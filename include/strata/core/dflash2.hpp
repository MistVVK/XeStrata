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
#include "strata/core/gpu.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"

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
    static constexpr int kRows = 8;           ///< block rows: the anchor + 7 masks
    static constexpr int kDrafts = kRows - 1;
    static constexpr int kRing = 2064;        ///< K/V slots per layer: the 2048 window + slack for rows written ahead of a rejection

    Dflash2Drafter() = default;
    ~Dflash2Drafter();
    Dflash2Drafter(const Dflash2Drafter&) = delete;
    Dflash2Drafter& operator=(const Dflash2Drafter&) = delete;

    /// Reads the pack (tools/dflash2_pack.py) to the GPU `device` and allocates the ring and the buffers.
    bool load(const std::string& pack_path, int device, std::string& err);
    /// The target's embedding and head (and the draft vocabulary file, "" or missing: the whole head).  The head's
    /// rows over the subset are gathered once (core/draft_head, the MTP drafter's shared part).
    bool bind(const WeightTable& wt, const ModelGeometry& g, const NativeHead* head, const std::string& vocab_file,
              std::string& err);
    /// The MTP drafter's gathered draft-head rows and vocabulary (it keeps owning them) when they are the same bytes
    /// this drafter would gather (the same head and vocabulary file): bind then uses them instead of a second copy.
    struct SharedHead { const uint8_t* rows = nullptr; const int32_t* vocab_dev = nullptr; int64_t n = 0; int type = -1; };
    void share_head(const SharedHead& s) { shared_ = s; }
    bool head_shared() const { return dhead_shared_; }
    /// The plain engine's / a test's embedding and head: raw callbacks (the head runs over the planar q8_1 rows this
    /// drafter quantizes, `head_rows` rows of the native form, or the gathered subset).
    using EmbedFn = std::function<bool(const int32_t* tokens_dev, int n, float* out, void* stream, std::string& err)>;
    using HeadFn = std::function<void(const uint8_t* rows, const uint8_t* xq, int n, float* y, void* stream)>;
    bool bind_raw(EmbedFn embed, HeadFn head, int head_type, int64_t head_rows, const std::vector<int32_t>& vocab,
                  std::string& err);

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
    struct RingStats {
        int64_t requests = 0, rebuilds = 0, rows_rebuilt = 0, rows_missing = 0, stash_hits = 0, stash_stores = 0,
                stash_evictions = 0, cold_starts = 0;
        double ms_rebuild = 0, ms_swap = 0;
    };
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
    /// One round: the new context rows (n_new in 1..8: the committed window rows, taps device [n_new][ntap * n_embd],
    /// positions pos0 ...), then the block for `anchor` (the last committed token, at pos0 + n_new).  `drafts` /
    /// `probs` get kDrafts entries (the selector's pick and its probability).  The taps pointer must be the same
    /// device buffer on every call (the graphs are captured on it).
    bool draft(const float* taps_dev, int n_new, int64_t pos0, int32_t anchor, int32_t* drafts, float* probs,
               std::string& err);
    /// Captures the eight round graphs (n_new = 1..8) on `taps` now, so no round pays for a capture.
    bool prepare(const float* taps_dev, std::string& err);
    /// The most drafts a window will use (the engine's widest window - 1, 1..kDrafts; default kDrafts).  The head,
    /// the top-k and the selector's walk run for that many rows only: the first `n` drafts and probabilities are the
    /// ones a full block makes (every row is its own column, the walk goes left to right); draft() returns token 0
    /// with probability 0 for the rest.  Before prepare() / the first draft(); a later change drops the graphs.
    void set_max_drafts(int n);

    uint64_t vram_bytes() const { return vram_; }
    int64_t n_draft_vocab() const { return n_dvocab_; }
    int n_embd() const { return N_; }
    int mask_token() const { return mask_token_; }
    const std::vector<int>& tap_layers() const { return taps_; }
    int device() const { return device_; }
    strata::gpu::Stream stream() const { return cs_; }
    /// Tests: the whole ring as it is now (layer-major: per layer K then V, each [8 heads][kRing slots][128]) and its
    /// slot tags (epoch << 32 | position, -1 none).
    bool debug_ring(std::vector<float>& kv, std::vector<int64_t>& tags, std::string& err);

    double ms_draft = 0, ms_ctx = 0, ms_capture = 0;
    int64_t rounds = 0;

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
    bool alloc(std::string& err);
    /// The host's staging of one launch: positions and rope tables (rows 0..7 context, 8..15 block), the anchor,
    /// then ONE copy to the device.
    void stage(int n_ctx, int64_t pos0, int64_t block_s, int32_t anchor);

    int device_ = -1;
    strata::gpu::Stream cs_ = nullptr;
    std::vector<Tensor> tensors_;
    uint8_t* blob_ = nullptr;
    uint8_t* slab_ = nullptr;       ///< one device allocation for the work buffers, the rings and the rebuild scratch
    uint64_t vram_ = 0;
    int N_ = 2560, L_ = 5, NH_ = 32, NKV_ = 8, HD_ = 128, I_ = 7680, window_ = 2048, group_ = 16, rank_ = 256,
        mask_token_ = 0;
    float theta_ = 10000.0f, eps_ = 1e-6f;
    std::vector<int> taps_;
    int64_t vocab_ = 0;
    bool sel_q4_ = false;           ///< the selector's codebooks are Q4_0 rows (the converted pack)

    // target pieces
    EmbedFn embed_;
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
    int32_t* d_meta_ = nullptr;     ///< ints [0..7] context positions, [8..15] block positions, [16] anchor token,
                                    ///< [17] anchor position, [18] epoch, [24..31] block tokens; then 16 rope tables of 128 floats
    int32_t* h_meta_ = nullptr;     ///< host staging of d_meta_ (host USM)
    float* rt_ctx_ = nullptr;       ///< d_meta_ + 32 (as floats); rt_blk_ = rt_ctx_ + 8 * 128
    float* rt_blk_ = nullptr;
    int32_t* d_out_ = nullptr;      ///< the select kernel's 7 picks, then 9 pad, then 7 probabilities
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
    strata::gpu::Graph* exec_[kRows + 1] = {};
    const float* graph_taps_ = nullptr;
    int ndraft_ = kDrafts;
    int64_t have_ctx_ = 0;
    bool ctx_pending_ = false;      ///< context_async queued rows nobody has waited for yet

    // ---- ring log (see begin_request)
    static constexpr int kLogChunk = 256;   ///< positions per log chunk (allocated on first use)
    struct LogChunk { std::vector<int8_t> q; std::vector<float> d; std::vector<uint8_t> ok; };
    struct Lineage {
        std::vector<int32_t> hist;                        ///< the token of every position the ring was filled for
        std::vector<std::shared_ptr<LogChunk>> chunks;    ///< the activation rows by position / 256 (copy on write)
        bool logged(int64_t pos) const;
        uint64_t bytes() const;
        void truncate(int64_t keep);
        LogChunk& writable(size_t chunk);
    };
    Lineage cur_;
    std::deque<Lineage> stash_;
    bool log_on_ = true;
    uint64_t stash_budget_ = 256ull << 20;
    int8_t* h_logq_ = nullptr;      ///< host USM: the last update's codes [8][2560] ...
    float* h_logd_ = nullptr;       ///< ... and scales [8][80], copied from the device after the update
    int64_t pend_pos0_ = -1;
    int pend_n_ = 0;
    int64_t slot_pos_[kRing] = {};  ///< host shadow of the device ring tags
    uint8_t* rb_q_ = nullptr;       ///< rebuild scratch (device): the window's codes, scales and positions
    float* rb_d_ = nullptr;
    int32_t* rb_pos_ = nullptr;
    void log_stage(int64_t pos0, int n);
    void log_fetch(int n);
    void log_flush();
    bool rebuild(int64_t lo, int64_t hi, int64_t* rebuilt, int64_t* missing, std::string& err);
    void stash_push(Lineage&& l);
    void invalidate_ring();
    int64_t ring_hi_ = 0;
    int32_t epoch_ = 0;
};

}  // namespace strata::core