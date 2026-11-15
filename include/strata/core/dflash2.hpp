// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/core/dflash2.hpp - the DFlash2 block drafter, for every engine (Qwen3.8-Flash-Next's today; the
// mechanism is the plain engine's too, over its own embedding and head).
//
// incoai/Qwen3.8-Flash-Next-DFlash2 (Apache-2.0) is a block drafter: 5 transformer layers of hidden 2560 that read
// the target's hidden states (the mean of the hyper-connection streams after the pack's tap layers) as context and,
// from the last accepted token plus 7 mask tokens, produce 7 draft tokens in one forward pass.  A candidate selector
// (the head's top-16 per row, a rank-256 bilinear score between the previous pick and each candidate) picks the path
// left to right.  tools/dflash2_pack.py packs it (Q4_0 or Q8_0 matrices, bf16 or Q4_0 selector codebooks).
//
//   context    taps (ntap x 2560 per accepted row) -> fc -> hidden_norm -> per layer K (norm + rope) and V -> ring slot pos % R
//   block      8 rows: embed(anchor), embed(mask) x 7; per layer
//                rms -> dynamic 2-tap conv (prepare) -> q, k, v -> q/k norm + rope -> attention over the ring + the block's own keys
//                -> o -> conv (finish) + residual -> rms -> conv (prepare) -> gate/up -> silu*up -> down -> conv (finish) + residual
//   selector   final norm -> head over the draft vocabulary (rows 1..7) -> top-16 -> walk
//
// The ring holds only COMMITTED rows (an accepted window row's taps arrive after the verify window), and the block's
// keys stay outside it, so a rejected draft leaves nothing to roll back.  One graph per context-row count n = 1..8
// holds the context update and the block; the host stages the positions, the anchor and the rope tables (computed on
// the host: no device FP64) and waits for the 7 drafts.  A draft only proposes: the verify window decides every
// emitted token, so the drafter changes the speed, never the answer.
#pragma once

#include "strata/core/draft_head.hpp"
#include "strata/core/draft_ring_log.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/kernels/window_limits.hpp"

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

class NativeHead;

class Dflash2Drafter {
public:
    static constexpr int kRows = strata::kernels::kVerifyNarrowT;           ///< block rows: the anchor + 7 masks
    static constexpr int kDrafts = kRows - 1;
    static constexpr int kMaxRows = strata::kernels::kWindowMaxT;
    static constexpr int kMaxDrafts = kMaxRows - 1;
    static constexpr int kRing = 2064;        ///< K/V slots per layer: the 2048 window + slack for rows written ahead of a rejection

    Dflash2Drafter() = default;
    ~Dflash2Drafter();
    Dflash2Drafter(const Dflash2Drafter&) = delete;
    Dflash2Drafter& operator=(const Dflash2Drafter&) = delete;

    /// A pack's tap layers, hidden size and vocabulary, and the bytes its tensors take on the GPU, from its index
    /// alone (nothing is loaded): an engine sets its taps before the drafter is loaded and checks the pack is its
    /// model's (the hidden size, the vocabulary, the taps inside the model's layers).
    struct PackShape { std::vector<int> taps; int hidden = 0; int64_t vocab = 0; uint64_t bytes = 0; };
    static bool read_shape(const std::string& pack_path, PackShape& shape, std::string& err);
    /// Reads the pack (tools/dflash2_pack.py) to the GPU `device` and allocates the ring and the buffers.
    bool load(const std::string& pack_path, int device, std::string& err);
    /// Set before load: reserve the wider buffers only when a caller will measure or use them.
    void set_capacity_rows(int rows);
    /// The rows its windows are sized for: what set_capacity_rows asked until load, then the load's
    int capacity_rows() const { return device_ >= 0 ? capacity_rows_ : requested_capacity_rows_; }
    /// The target's embedding and head (and the draft vocabulary file, "" or missing: the whole head).  The head's
    /// rows over the subset are gathered once (core/draft_head, the MTP drafter's shared part).
    bool bind(const WeightTable& wt, const ModelGeometry& g, const NativeHead* head, const std::string& vocab_file,
              std::string& err);
    /// The MTP drafter's gathered draft-head rows and vocabulary (it keeps owning them) when they are the same bytes
    /// this drafter would gather (the same head and vocabulary file): bind then uses them instead of a second copy.
    struct SharedHead { const uint8_t* rows = nullptr; const int32_t* vocab_dev = nullptr; int64_t n = 0; int type = -1; };
    void share_head(const SharedHead& s) { shared_ = s; }
    bool head_shared() const { return dhead_shared_; }
    /// A test's embedding and head: raw callbacks (the head runs over the q8_1 rows this drafter quantizes - `xq`,
    /// or the same rows in fp32, `xf`, for a head of bf16 rows - `head_rows` rows of the native form, or the gathered
    /// subset).
    using EmbedFn = std::function<bool(const int32_t* tokens_dev, int n, float* out, void* stream, std::string& err)>;
    using HeadFn = std::function<void(const uint8_t* rows, const uint8_t* xq, const float* xf, int n, float* y,
                                      void* stream)>;
    bool bind_raw(EmbedFn embed, HeadFn head, int head_type, int64_t head_rows, const std::vector<int32_t>& vocab,
                  std::string& err);
    /// The plain engine's: its embedding is dequantized on the host a token at a time (the table stays in the model's
    /// mapping), so the drafter asks for two rows only - the mask token's once here, the anchor's as it stages a
    /// round - and the round's graph copies them into the block.  `head_rows` rows of the head, `vocab_dev` (null:
    /// the whole head) the token of each row, the engine's (it keeps owning it).
    using HostEmbedFn = std::function<bool(int32_t token, float* row)>;
    bool bind_host(HostEmbedFn embed, HeadFn head, int64_t head_rows, const int32_t* vocab_dev, std::string& err);
    /// Everything this drafter holds on the GPU and in RAM given back (the pack, the work buffers and the rings, the
    /// draft head and logits it owns, the ring log and its stash, the stream and the graphs): the engine's startup
    /// measurement drops the drafter that lost.  It is unloaded after this (load again to use it).
    void release();
    bool loaded() const { return blob_ != nullptr; }
    /// The host memory the drafter holds (the staging and the ring log with its stash).
    uint64_t ram_bytes() const;

    /// A new request: the ring is empty again.
    void reset();
    /// Serve: a request starts with prompt `ids` (n tokens) and the session continues at `resume`.  The drafter keeps
    /// the token history of the rows its ring was filled for, and (the ring log) the activation rows those K/V rows
    /// were made from.  The window the first draft reads is positions n - 2047 ... n - 1; the part below `resume` is
    /// not read again by the prompt, so it must already be in the ring: a lineage whose history agrees with
    /// ids[0, resume) has its overwritten slots rebuilt from the log; a parked lineage (the stash) that agrees is
    /// swapped in; neither: the ring starts over and `*cold` is set (the drafts are poor for the first rounds, never
    /// wrong: the verify decides).  `*cold` is also set when some of the window's rows were never logged.
    bool begin_request(const int32_t* ids, int64_t n, int64_t resume, bool* cold);
    /// The ring log (XESTRATA_DF2_RING_LOG=0 turns it off, the pre-log behaviour for A/Bs; XESTRATA_DF2_LOG_MIB sets
    /// the stash, default 256): every context update copies the quantized activation rows the K/V of the layers are
    /// made from (2560 int8 codes + 80 block scales a position, 2.9 KB) into a host log of the lineage; a rebuild runs
    /// the K/V projections, norms, rope and ring writes over the logged rows with the same kernels, so the rebuilt
    /// K/V bytes are those of the original update.
    using RingStats = DraftRingHistory::Stats;
    RingStats ring_stats;
    int64_t last_rebuilt = 0, last_missing = 0;
    double last_ms = 0;
    bool last_swapped = false;
    bool ring_log_on() const { return log_on_; }
    uint64_t ring_log_bytes() const;
    uint64_t ring_stash_bytes() const;
    /// Serve: rows [pos0, pos0 + n) were committed with these tokens (a prompt chunk, a window's accepted rows).
    void note_rows(const int32_t* toks, int64_t n, int64_t pos0);
    /// Context rows without a draft: `n` rows of taps (device, [n][ntap * n_embd]) for positions pos0 ... pos0 + n - 1.
    bool context(const float* taps_dev, int64_t n, int64_t pos0, std::string& err);
    /// The same without the wait: the rows are queued on the drafter's stream and a later context_wait() (or the next
    /// draft / context call, which waits first) finds them done.  --draft auto keeps the idle drafter current this way.
    bool context_async(const float* taps_dev, int64_t n, int64_t pos0, std::string& err);
    bool context_wait(std::string& err);
    /// One round: the new context rows (n_new in 1..capacity: the committed window rows, taps device
    /// [n_new][ntap * n_embd], positions pos0 ...), then the block for `anchor` (the last committed token, at
    /// pos0 + n_new). `drafts` / `probs` each hold the selected block's rows - 1 entries. The taps pointer must be the same
    /// device buffer on every call (the graphs are captured on it).
    bool draft(const float* taps_dev, int n_new, int64_t pos0, int32_t anchor, int32_t* drafts, float* probs,
               std::string& err);
    /// Captures the round graphs on `taps` now. With 15 drafts enabled this captures both block shapes and context counts.
    bool prepare(const float* taps_dev, std::string& err);
    /// The most drafts a window will use (1..capacity - 1, default 7). The head and selector stop at that many rows.
    /// The draft block itself is non-causal: changing its shape can change even the first seven drafts. A later change
    /// drops the graphs.
    void set_max_drafts(int n);
    /// Select a captured shape. The caller supplies arrays of rows - 1 entries.
    void set_block_rows(int rows);

    uint64_t vram_bytes() const { return vram_; }
    /// The VRAM bind() will allocate for the draft head over `vocab_file`'s subset (or the whole head): the gathered
    /// rows and the draft logits.  The expert cache is sized before bind(), so it reserves this.
    uint64_t bind_bytes(uint64_t head_row_bytes, int64_t n_vocab, const std::string& vocab_file) const;
    int64_t n_draft_vocab() const { return n_dvocab_; }
    int64_t vocab() const { return vocab_; }
    int n_embd() const { return N_; }
    int mask_token() const { return mask_token_; }
    const std::vector<int>& tap_layers() const { return taps_; }
    int device() const { return device_; }
    strata::gpu::Stream stream() const { return cs_; }
    /// Tests: the whole ring as it is now (layer-major: per layer K then V, each [8 heads][kRing slots][128]) and its
    /// slot tags (epoch << 32 | position, -1 none).
    bool debug_ring(std::vector<float>& kv, std::vector<int64_t>& tags, std::string& err);

    double ms_draft = 0, ms_ctx = 0, ms_capture = 0;
    int64_t rounds = 0, rounds_short = 0, rounds_wide = 0;

private:
    struct Tensor { std::string name, dtype; int64_t rows = 0, cols = 0; uint64_t off = 0, bytes = 0; };
    const Tensor& find(const std::string& name) const;
    const float* f32(const std::string& name) const { return reinterpret_cast<const float*>(blob_ + find(name).off); }
    /// The matrices of one launch (1..3, same input width) times the planar activations xq / xd (ncols columns).
    bool gemm(std::initializer_list<const char*> names, std::initializer_list<float*> outs, int ncols,
              const uint8_t* xq, const float* xd, std::string& err);
    bool record_context(const float* taps, int n, std::string& err);
    bool record_kv(int n, const uint8_t* xq, const float* xd, const int32_t* pos, std::string& err);
    bool record_block(std::string& err);
    bool capture(int n, const float* taps, std::string& err);
    /// The block's input rows into h_: the target's embedding of the anchor and the mask tokens
    bool embed_rows(std::string& err);
    bool alloc(std::string& err);
    /// Stage the context and block positions, rope tables, and anchor (and its host embedding) in one device copy.
    void stage(int n_ctx, int64_t pos0, int64_t block_s, int32_t anchor, bool block = true);
    int meta_ints() const { return capacity_rows_ == kRows ? 32 : 64; }
    int table_floats() const { return capacity_rows_ * 2 * 128; }
    size_t meta_bytes() const { return (size_t) (meta_ints() + table_floats()) * 4; }
    int block_pos_offset() const { return capacity_rows_ == kRows ? 8 : 32; }
    int token_offset() const { return capacity_rows_ == kRows ? 24 : 48; }

    int device_ = -1;
    strata::gpu::Stream cs_ = nullptr;
    std::vector<Tensor> tensors_;
    uint8_t* blob_ = nullptr;
    uint8_t* slab_ = nullptr;       ///< one device allocation for the work buffers, the rings and the rebuild scratch
    uint64_t vram_ = 0;
    int N_ = 2560, L_ = 5, NH_ = 32, NKV_ = 8, HD_ = 128, I_ = 7680, window_ = 2048, group_ = 16, rank_ = 256,
        mask_token_ = 0;
    int in_max_ = 12800;            ///< the widest product input (the taps' fc, the FFN's down, the attention's o)
    float theta_ = 10000.0f, eps_ = 1e-6f;
    std::vector<int> taps_;
    int64_t vocab_ = 0;
    bool sel_q4_ = false;           ///< the selector's codebooks are Q4_0 rows (the converted pack)

    // target pieces
    EmbedFn embed_;
    HostEmbedFn host_embed_;        ///< bind_host: the anchor's row staged with the round (h_meta_), the mask's in emb_mask_
    HeadFn head_fn_;
    const uint8_t* head_w_ = nullptr;
    int head_type_ = -1;
    int64_t n_dvocab_ = 0;
    const int32_t* dvocab_ = nullptr;
    std::shared_ptr<DraftSubset> subset_;   ///< the gathered draft head (owned), or the shared one
    SharedHead shared_;
    bool dhead_shared_ = false;
    bool bound_ = false;

    // device buffers (the slab)
    int32_t* d_meta_ = nullptr;     ///< context positions, anchor at 16, position 17, epoch 18, block positions/tokens;
                                    ///< layout offsets and rope-table count depend on capacity_rows_
    int32_t* h_meta_ = nullptr;     ///< host staging of d_meta_ (host USM); then, with bind_host, the anchor's row
    float* emb_mask_ = nullptr;     ///< bind_host: the mask token's embedding row
    bool vocab_borrowed_ = false;   ///< bind_host's vocab_dev: the engine's, not freed here
    float* rt_ctx_ = nullptr;       ///< d_meta_ + meta_ints(); rt_blk_ follows capacity_rows_ rope tables
    float* rt_blk_ = nullptr;
    int32_t* d_out_ = nullptr;      ///< up to 15 picks, then probabilities at float offset 16
    int32_t* h_out_ = nullptr;
    float* kring_[8] = {};
    float* vring_[8] = {};
    uint8_t* xq_ = nullptr;         ///< the head's q8_1 activations
    uint8_t* xqp_ = nullptr;        ///< planar activations of the block's GEMMs
    float* xdp_ = nullptr;
    uint8_t* xqc_ = nullptr;        ///< the same for the context update
    float* xdc_ = nullptr;
    float *fcy_ = nullptr, *kc_ = nullptr, *vc_ = nullptr;
    float *h_ = nullptr, *xn_ = nullptr, *dyn_ = nullptr, *q_ = nullptr, *qn_ = nullptr, *k_ = nullptr, *v_ = nullptr,
          *kblk_ = nullptr;
    float *o_ = nullptr, *gate_ = nullptr, *up_ = nullptr, *mlp_ = nullptr, *part_ = nullptr, *fn_ = nullptr;
    float *logits_ = nullptr, *topval_ = nullptr, *hsel_ = nullptr;
    int32_t* toptok_ = nullptr;
    uint32_t* topscratch_ = nullptr;
    int64_t* tag_ = nullptr;
    strata::gpu::Graph* exec_[2][kMaxRows + 1] = {};
    const float* graph_taps_ = nullptr;
    int ndraft_ = kDrafts;
    int block_rows_ = kRows;
    int capacity_rows_ = kRows;
    int requested_capacity_rows_ = kRows;
    int64_t have_ctx_ = 0;
    bool ctx_pending_ = false;      ///< context_async queued rows nobody has waited for yet

    // ---- ring log (see begin_request)
    DraftRingHistory history_;
    bool log_on_ = true;
    uint64_t stash_budget_ = 256ull << 20;
    int8_t* h_logq_ = nullptr;      ///< host USM: the last update's codes [8][2560] ...
    float* h_logd_ = nullptr;       ///< ... and scales [8][80], copied from the device after the update
    int64_t pend_pos0_ = -1;
    int pend_n_ = 0;
    uint8_t* rb_q_ = nullptr;       ///< rebuild scratch (device): the window's codes, scales and positions
    float* rb_d_ = nullptr;
    int32_t* rb_pos_ = nullptr;
    void log_stage(int64_t pos0, int n);
    void log_fetch(int n);
    void log_flush();
    bool replay(const int32_t* positions, const int8_t* q, const float* scales, int rows, int32_t epoch,
                std::string& err);
};

}  // namespace strata::core