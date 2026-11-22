// mla_attn.hip.h - Hy4-Flash MLA attention kernels for gfx1151 (wave32, RDNA 3.5), MQA-latent (absorbed) form.
//
// Model geometry (config of Hy4-Flash): kv_lora 512, qk_nope 192, qk_rope 64 (interleaved rope pairs 2i,2i+1), qk_head 256, v_head 256,
// 32 heads on the 12 sparse layers (theta 1e7), 64 heads on the 30 sliding-window layers (theta 1e4, window 512). One latent KV head.
//
// KV cache row (576 halves) = [ kv_cmpr (512, RMSNormed * kv_a_layernorm.weight) | k_pe (64, rope-rotated) ]. The value of a key is kv_cmpr (first 512).
// CACHE LAYOUT (one buffer per layer, allocate mla_cache_bytes(slots) bytes, slots % 16 == 0, ZERO-INITIALISED once with hipMemset):
//     rows : slots x 576 halves  (the layout above, slot s at cache + s*576). Total 1152 B per slot (no duplicate Vt; PV transposes in LDS/shared memory).
//   Ring (SWA layers): slots = ring = 512 + chunk_max rounded up to a multiple of 16, slot = position % ring.
// Per head h: kv_b_proj rows are 448 per head (192 k_nope rows, then 256 v rows), cols = 512, so
//     wk_b[h][i][c] = Wkvb[h*448 + i][c]        (i < 192)       -> absorbed into the query:  qa[0:512][c] = sum_i q_nope[i] * wk_b[h][i][c]
//     wv_b[h][j][c] = Wkvb[h*448 + 192 + j][c]  (j < 256)       -> applied after the attention:  o[j] = sum_c wv_b[h][j][c] * ctx[c]
// score(q, key) = (qa . row[0:512] + rope(q_pe) . row[512:576]) / sqrt(256)   (scale = 1/16; NOT 1/sqrt(576))
// softmax over the keys plus ONE per-head learnable sink logit (joins the denominator, has no value).  o *= sigmoid(gate) when a gate is passed.
//
// All functions take a hipStream_t first, device pointers, and are asynchronous on that stream (no allocation inside, no sync).
#pragma once
#include "hy4_gpu.h"

#include <cstddef>
#include <cstdint>

namespace hy4 {

constexpr int MLA_KV_LORA = 512;
constexpr int MLA_ROPE = 64;
constexpr int MLA_ROW = 576;     // halves per FP16 cache row (1152 B/slot)
// Compressed Q8 MLA cache row (656 B/slot = 328 halves, 43.1% smaller than FP16, 16-byte aligned):
//   [0..511]   int8_t kv_cmpr[512] (8 groups of 64, symmetric INT8)
//   [512..527] __half sc[8]        (per-64-element scale = amax / 127.0f)
//   [528..655] __half k_pe[64]     (lossless FP16 RoPE key!)
constexpr int MLA_ROW_Q8_BYTES = 656;
constexpr int MLA_ROW_Q8_HALVES = 328;
constexpr int MLA_NOPE = 192;
constexpr int MLA_QHEAD = 256;   // q_b output per head: [192 nope | 64 pe]
constexpr int MLA_VHEAD = 256;

enum MlaMode : int {
    MLA_CAUSAL = 0,  // query at position p attends cache positions [0 .. p]            (cache slot = position, or position % ring when ring > 0)
    MLA_WINDOW = 1,  // positions (p - window, p], slot = position % ring (ring > 0 required, ring >= window + M)
    MLA_LIST = 2     // explicit per-query list of CACHE SLOT indices: keys[t*kmax + i], i < nkeys[t]   (IndexPool picks; no ordering needed)
};

struct TreePos {
    int n = 0;             // 0 = standard linear positions p0 + t; 1..8 = tree verify positions p0 + depth[t]
    uint8_t depth[8] = {}; // logical depth of each tree node (or source node indices for compaction)
};

struct MlaAttnArgs {
    int M = 1;              // number of queries (chunk of tokens), query t sits at position p0 + t (or p0 + tree_depth[t] when tree_nodes > 0)
    int NH = 32;            // 32 (sparse layers) or 64 (SWA layers)
    int p0 = 0;             // position of the first query of the chunk
    int mode = MLA_CAUSAL;
    int window = 512;       // MLA_WINDOW
    int ring = 0;           // ring size (slots) of the cache; 0 = linear cache (slot = position)
    int cache_slots = 0;    // total slots of the cache buffer (the Vt region starts at cache + cache_slots*576 halves); must equal the value given to mla_write_kv
    bool kv_q8 = false;     // true: compressed Q8 MLA KV cache (656 B/slot: int8 kv_cmpr[512] + half sc[8] + lossless FP16 k_pe[64])
    // MLA_LIST (IndexPool picks). Entries are consumed in blocks of 4: entry 4b+u is the slot of position 4*G+u of ONE aligned 4-token group G
    // (for the fast path the 4 entries of a block must be the slots 4G..4G+3 in order; the last block may be partial, nkeys counts valid entries).
    const int32_t* keys = nullptr;   // [M][kmax] cache slot indices (logical positions when tree_nodes > 0), kmax % 4 == 0
    const int32_t* nkeys = nullptr;  // [M]
    int kmax = 0;
    float scale = 0.0625f;  // 1/sqrt(256)
    const int* p0_dev = nullptr;  // optional device pointer to step-varying p0 (for CUDA/HIP Graph replay)
    // Tree-8 speculative verification (tree_nodes == 0: standard linear chunk; 1..8: tree node t sits at logical position p0 + tree_depth[t]
    // and its ancestor at logical position p0 + d is stored in physical cache slot (p0 + tree_anc[t][d]) [% ring])
    struct TreeDev {
        int tree_nodes = 0;
        uint8_t tree_depth[8] = {};
        uint8_t tree_anc[8][8] = {};
    };
    int tree_nodes = 0;
    uint8_t tree_depth[8] = {};
    uint8_t tree_anc[8][8] = {};
    const TreeDev* tree_dev = nullptr;  // optional device pointer to step-varying tree topology (CUDA/HIP Graph replay)
};

// ---- host helpers (not device code) --------------------------------------------------------------------------------------------------------------------
// bytes of workspace needed by mla_attention() (the fp32 ctx buffer [M][NH][512] plus split-K partials for the decode path)
size_t mla_workspace_bytes(int M, int NH, int max_keys);
size_t mla_workspace_bytes(const struct MlaAttnArgs& a);   // exact for this call (derives max_keys from the mode; use this one)
int mla_choose_splits(const struct MlaAttnArgs& a);        // exact split count S that mla_attend will use for these args
// bytes of one layer's KV cache buffer with `slots` slots (FP16: 1152 B/slot, Q8: 656 B/slot)
size_t mla_cache_bytes(int slots, bool kv_q8 = false);

// ---- kernels -------------------------------------------------------------------------------------------------------------------------------------
// 1. q absorb + rope.   q_b: fp32 [M][NH][256] (output of the q_b_proj GEMM; first 192 = nope, last 64 = pe).
//    qa: half [M][NH][576] = [ sum_i q_nope[i] * wk_b[h][i][:]  (512) | rope(q_pe, position p0+t, theta) (64) ].   wk_b: half [NH][192][512].
void mla_prep_q(hipStream_t s, const float* q_b, const __half* wk_b, __half* qa, int M, int NH, int p0, float theta);

// 1b. matrix-core version of the same q absorb (use for prefill chunks / verify windows; ~M >= 8).  wk_bt: half [NH][512][192] = wk_b transposed
//     (build it once with mla_transpose_wk; wk_b may be freed afterwards).  Same outputs as mla_prep_q.
void mla_transpose_wk(hipStream_t s, const __half* wk_b, __half* wk_bt, int NH);
void mla_prep_q_t(hipStream_t s, const float* q_b, const __half* wk_bt, __half* qa, int M, int NH, int p0, float theta);

// 1c. DECODE variants (M = 1..8, e.g. decode and DFly2 verify windows) that read the QUANTIZED kv_b_proj tensor directly (S_K wire format, fmt = 2,3,4,5,6,8 as FMT_S*,
//     [NH*448][512], 2 super-blocks per row): no fp16 absorbed copies are needed for decode and ~3x fewer weight bytes are read per token.
//     mla_prep_q_dec == mla_prep_q (q absorb + rope), mla_vup_dec == mla_vup (v up-projection + gate), given the raw kv_b_proj bytes on the device.
void mla_prep_q_dec(hipStream_t s, int fmt, const void* kv_b, const float* q_b, __half* qa, int M, int NH, int p0, float theta, TreePos tp = {});
void mla_prep_q_dec_d(hipStream_t s, int fmt, const void* kv_b, const float* q_b, __half* qa, int M, int NH, const int* p0_dev, float theta, const TreePos* tp_dev = nullptr);
void mla_vup_dec(hipStream_t s, int fmt, const void* kv_b, const __half* ctx, const float* gate, __half* o, int M, int NH);

// 2. KV cache write.  kv_a: fp32 [M][576] (output of kv_a_proj_with_mqa: 512 latent | 64 pe). Writes cache slot (p0+t) % ring (ring = 0: slot = p0+t):
//    cache[slot][0:512] = half(kv_a[t][0:512] * rsqrt(mean(kv_a[t][0:512]^2) + eps) * norm_w[c]);  cache[slot][512:576] = rope(kv_a[t][512:576], pos, theta).
void mla_write_kv(hipStream_t s, const float* kv_a, const float* norm_w, __half* cache, int cache_slots, int M, int p0, int ring, float theta, float eps, TreePos tp = {}, bool kv_q8 = false);
void mla_write_kv_d(hipStream_t s, const float* kv_a, const float* norm_w, __half* cache, int cache_slots, int M, const int* p0_dev, int ring, float theta, float eps, const TreePos* tp_dev = nullptr, bool kv_q8 = false);

// 2b. Compact accepted tree-branch KV rows: for k = 0..src.n-1 copies physical slot (p0 + src.depth[k]) -> (p0 + k) (mod ring if ring > 0).
//     Requires src.depth[k] >= k (topological order), so sequential in-place compaction is race-free.
void mla_compact_kv(hipStream_t s, __half* cache, int cache_slots, int p0, int ring, TreePos src, bool kv_q8 = false);

// 3. attention, latent output.  qa: half [M][NH][576]; cache: half rows; sinks: fp32 [NH]; ctx: HALF [M][NH][512] (softmax-weighted latent rows).
//    The cache must already hold every key position the chunk can see (write the chunk's own K/V first).  ws: scratch (>= mla_workspace_bytes()), may be null for tiny M*keys.
void mla_attend(hipStream_t s, const __half* qa, const __half* cache, const float* sinks, __half* ctx, const MlaAttnArgs& a, void* ws, size_t ws_bytes);

// 4. v up-projection (+ output gate).  ctx: half [M][NH][512]; wv_b: half [NH][256][512]; gate: fp32 [M][NH*256] (raw logits, sigmoid applied here) or nullptr;
//    o: half [M][NH*256] ready for the o_proj GEMM.
void mla_vup(hipStream_t s, const __half* ctx, const __half* wv_b, const float* gate, __half* o, int M, int NH);

// convenience: attend + vup.  ctx_ws: scratch of M*NH*512 floats (the latent context is kept as half inside, so M*NH*512*2 bytes are used); ws as in mla_attend.
void mla_attention(hipStream_t s, const __half* qa, const __half* cache, const float* sinks, const __half* wv_b, const float* gate, __half* o, float* ctx_ws,
                   const MlaAttnArgs& a, void* ws, size_t ws_bytes);

// forces the simple reference-quality kernels (for tests / bisecting); default 0 = fast paths when available
extern int g_mla_force_simple;
extern int g_mla_target_blocks;  // decode split-K tuning: aim for this many blocks when M*NH/16 is small (default 192)
extern int g_mla_batch_invariant;  // 1: for M <= 16 the split count is chosen as for M = 1 (fixed key partition per kv length: rows of a verify window sum in the same order as one-token decode)

}  // namespace hy4
