// include/strata/core/dflash2.hpp - the DFlash2 drafter for Qwen3.8-Flash-Next on the GPU (CUDA and HIP).
//
// incoai/Qwen3.8-Flash-Next-DFlash2 (Apache-2.0) is a block drafter: 5 transformer layers of hidden 2560 that read the target's hidden
// states (the mean of the 4 hyper-connection streams after layers 5, 15, 25, 35, 45) as context and, from the last accepted token plus
// 7 mask tokens, produce 7 draft tokens in one forward pass.  A candidate selector (the head's top-16 per row, a rank-256 bilinear
// score between the previous pick and each candidate) picks the path left to right.  tools/dflash2_pack.py packs it as Q8_0.
//
//   context    taps (5 x 2560 per accepted row) -> fc (12800 -> 2560) -> hidden_norm -> per layer K (norm + rope) and V -> ring slot pos % R
//   block      8 rows: embed(anchor), embed(mask) x 7; per layer
//                rms -> dynamic 2-tap conv (prepare) -> q, k, v -> q/k norm + rope -> attention over the ring + the block's own keys
//                -> o -> conv (finish) + residual -> rms -> conv (prepare) -> gate/up -> silu*up -> down -> conv (finish) + residual
//   selector   final norm -> head over the draft vocabulary (rows 1..7) -> top-16 -> walk
//
// The ring holds only COMMITTED rows (an accepted window row's taps arrive after the verify window), and the block's keys stay outside
// it, so a rejected draft leaves nothing to roll back.  One CUDA graph per context-row count n = 1..8 holds the context update and the
// block; the host stages the positions and the anchor and waits for the 7 drafts.
#pragma once
#include <cstdlib>

#include "strata/core/session.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace strata::core {

class NativeHead;

class Dflash2Drafter {
public:
    static constexpr int kRows = 8;           ///< block rows: the anchor + 7 masks
    static constexpr int kDrafts = kRows - 1;
    static constexpr int kRing2048 = 2048;    ///< positions the deferred-rows buffer holds
    static constexpr int kRing = 2064;        ///< K/V slots per layer: the 2048 window + slack for rows written ahead of a rejection

    Dflash2Drafter() = default;
    ~Dflash2Drafter();
    Dflash2Drafter(const Dflash2Drafter&) = delete;
    Dflash2Drafter& operator=(const Dflash2Drafter&) = delete;

    /// Reads the pack (tools/dflash2_pack.py) to the GPU `device` and allocates the ring and the buffers (about 0.8 GB).
    bool load(const std::string& pack_path, int device, std::string& err);
    /// The target's embedding and head (and the draft vocabulary file, "" or missing: the whole head).
    /// `head_device` >= 0 and not this card's: the head sits on another card (a layer split's last stage); the draft vocabulary's rows are gathered there and
    /// copied over once, so the drafter can sit on the first stage's card.
    bool bind(const WeightTable& wt, const NativeHead* head, const std::string& vocab_file, std::string& err, int head_device = -1);
    /// STRATA_DF2_HEAD=native (CUDA / HIP): bind() ignores the vocabulary file and runs the draft head on the target's whole head (no pruned copy).
    static bool head_native() { const char* e = std::getenv("STRATA_DF2_HEAD"); return e != nullptr && e[0] == 'n'; }
    /// DF-05: before bind, the MTP drafter's gathered draft-head rows and vocabulary (device pointers it keeps owning) when they are the same bytes this
    /// drafter would gather; bind then uses them instead of a second copy (about 213 MiB on Flash-Next) and checks the vocabulary against the file.
    struct SharedHead { const uint8_t* rows = nullptr; const int32_t* vocab_dev = nullptr; const std::vector<int32_t>* vocab_host = nullptr; int64_t n = 0; int type = -1; };
    void share_head(const SharedHead& s) { shared_ = s; }
    bool head_shared() const { return dhead_shared_; }
    /// Tests: an embedding callback and a head given as raw rows of `head_type` (a native_mmvq type).
    using EmbedFn = std::function<void(const int32_t* tokens_dev, int n, float* out, cudaStream_t)>;
    bool bind_raw(EmbedFn embed, const void* head_w, int head_type, int64_t head_rows, const std::vector<int32_t>& vocab, std::string& err);

    /// Batch slots and layer splits.  A slot's drafter shares this one's immutable pack, embedding callback and draft head (same card) and owns only
    /// its ring, buffers, streams and graphs (about 100 MiB): `clone_alloc` after the parent's `load`, `clone_bind` after its `bind`, then `set_max_drafts` and `prepare` as for the parent.
    bool clone_alloc(const Dflash2Drafter& parent, std::string& err);   ///< after the parent's load: streams and buffers (before the expert cache is sized)
    bool clone_bind(const Dflash2Drafter& parent, std::string& err);    ///< after the parent's bind: the shared head, embedding and the draft logits
    /// Copy another drafter's context (ring, tags, token history, epoch) exactly: a request that moves between the solo session and a batch slot.  Same card.
    bool copy_state_from(Dflash2Drafter& src, std::string& err);
    /// A fixed device buffer [kRows][taps][n_embd] on this drafter's card: where a layer split's stages land their taps (Verifier::gather_taps) and the
    /// graphs are captured on it.  Allocated at load.
    float* landing() const { return landing_; }
    int device() const { return device_; }
    int n_taps() const { return (int) taps_.size(); }
    int64_t ring_hi() const { return ring_hi_; }

    /// DEFERRED CONTEXT (batch slots): a round with no spare window rows needs no draft, so a slot's drafter skips its forward work for that round.  The committed
    /// rows' taps wait in a pinned host buffer instead (last 2048 positions of one contiguous run, 51 KB a row, allocated at the first deferral); catch_up replays
    /// them through context() (the same kernels, in order, so the ring ends up as if every row had been taken at once) when spare rows return.
    /// defer_rows returns where to write `n` rows of taps for positions pos0 ... (a position that does not follow the held run starts a new one); the caller
    /// fills them (and waits for the copy) before the next catch_up.
    float* defer_rows(int64_t pos0, int n, std::string& err);
    bool has_deferred() const { return dpend_hi_ > dpend_lo_; }
    int64_t deferred_hi() const { return dpend_hi_; }
    const float* deferred_row(int64_t pos) const { return dpend_ + (size_t) (pos % kRing2048) * (size_t) (taps_.size() * (size_t) N_); }
    /// Replays the held rows below `upto` (at most the last 2047 of them) into the ring and drops them from the run (rows from `upto` on stay held).
    bool catch_up(int64_t upto, std::string& err);
    void clear_deferred() { dpend_lo_ = dpend_hi_ = 0; }
    int64_t n_catchups = 0, n_catchup_rows = 0, n_deferred_rows = 0;
    double ms_catchup = 0;

    /// A new request: the ring is empty again.
    void reset();
    /// Serve: a request starts with prompt `ids` (n tokens) and the session continues at `resume`.  The drafter keeps the token history of the rows its ring
    /// was filled for, and (RING LOG, below) the activation rows those K/V rows were made from.  The window the first draft will read is the n - 2047 ... n - 1
    /// positions; the part of it below `resume` is not read again by the prompt, so it must already be in the ring:
    ///   - the live lineage's history agrees with ids[0, resume) (a conversation going on, a repeated or rewound prefix): the ring slots a later, longer turn
    ///     overwrote (the ring wraps every 2064 positions) are rebuilt from the log;
    ///   - a parked lineage (kept in the stash when another conversation took over) agrees: it is swapped in and the window rebuilt from its log;
    ///   - neither: the ring starts over and `*cold` is set (the drafts are poor for the first rounds, never wrong: the verify decides).
    /// `*cold` is also set when some of the window's rows could not be rebuilt (never logged).
    bool begin_request(const int32_t* ids, int64_t n, int64_t resume, bool* cold);
    /// RING LOG (STRATA_DF2_RING_LOG=0 turns it off, which restores the pre-log behaviour for A/Bs).  Every context update copies the activation rows the K/V of all
    /// five layers are computed from (the int8 codes and block scales after hidden_norm: 2560 + 320 bytes a position, 2.9 KB, against 40 KB of K/V and 51 KB of taps)
    /// into a host log of the lineage; a rebuild runs the layers' K/V projections, norms, rope and ring writes over the logged rows, with the same kernels, so the
    /// rebuilt K/V bytes are those of the original update.  A lineage replaced by another is moved (not copied) into a stash of STRATA_DF2_LOG_MIB (default 256) MiB.
    struct RingStats {
        int64_t requests = 0, rebuilds = 0, rows_rebuilt = 0, rows_missing = 0, stash_hits = 0, stash_stores = 0, stash_evictions = 0, cold_starts = 0;
        double ms_rebuild = 0, ms_swap = 0;
    };
    RingStats ring_stats;
    int64_t last_rebuilt = 0, last_missing = 0;   ///< the last begin_request: ring rows rebuilt, rows that could not be
    double last_ms = 0;
    bool last_swapped = false;                    ///< ... and whether a parked lineage was brought back
    bool ring_log_on() const { return log_on_; }
    /// Allocates the log's pinned staging and turns the log on (STRATA_DF2_RING_LOG=0 keeps it off). alloc() calls it; a clone that skips alloc() calls it after its
    /// own allocation, on its device. Without it the log stays off and begin_request uses the pre-log rule.
    void init_ring_log();
    /// A request moves between two drafters whose rings were copied on the device (solo <-> a --parallel slot): the host side goes with it. The lineage's log is
    /// shared copy-on-write, the tag shadow and the epoch are copied (the caller copies kring_/vring_/tag_ on the device).
    void copy_log_state_from(const Dflash2Drafter& src);
    uint64_t ring_log_bytes() const;     ///< host bytes of the live lineage's log
    uint64_t ring_stash_bytes() const;   ///< host bytes of the parked lineages
    uint64_t ring_scratch_bytes() const { return rb_bytes_; }   ///< VRAM of the rebuild scratch (0 until the first rebuild)
    /// Serve: rows [pos0, pos0 + n) were committed with these tokens (a prompt chunk, a prompt window, a decode window's accepted rows).
    void note_rows(const int32_t* toks, int64_t n, int64_t pos0);
    /// Context rows without a draft: `n` rows of taps (device, [n][5 * n_embd]) for positions pos0 ... pos0 + n - 1.
    bool context(const float* taps_dev, int64_t n, int64_t pos0, std::string& err);
    /// The same without the wait: the rows are queued on the drafter's stream (n <= kRows) and a later context_wait() (or the next draft / context call, which
    /// waits first) finds them done.  The taps must stay as they are until then.  --draft auto keeps the idle drafter current this way, overlapped with the other's draft.
    bool context_async(const float* taps_dev, int64_t n, int64_t pos0, std::string& err);
    bool context_wait(std::string& err);
    /// One round: the new context rows (n_new in 1..8: the committed window rows, taps device [n_new][5 * n_embd], positions pos0 ...),
    /// then the block for `anchor` (the last committed token, at pos0 + n_new).  `drafts` / `probs` get kDrafts entries (the
    /// selector's pick and its probability).  The taps pointer must be the same device buffer on every call (graphs are captured on it).
    bool draft(const float* taps_dev, int n_new, int64_t pos0, int32_t anchor, int32_t* drafts, float* probs, std::string& err);

    /// draft() in two halves: draft_launch queues the round's graph and returns, draft_finish waits and reads the picks.  Several drafters (batch slots, each on its
    /// own stream) can launch first and finish after, so their graphs run at the same time.
    bool draft_launch(const float* taps_dev, int n_new, int64_t pos0, int32_t anchor, std::string& err);
    bool draft_finish(int32_t* drafts, float* probs, std::string& err);

    /// Captures the eight round graphs (n_new = 1..8) on `taps` now, so no round pays for a capture.
    bool prepare(const float* taps_dev, std::string& err);

    /// The most drafts a window will use (the engine's widest window - 1, 1..kDrafts; default kDrafts).  The head, the top-k and the selector's walk run for that
    /// many rows only: the first `n` drafts and probabilities are bitwise the ones a full block makes (every row is its own column, the walk goes left to right);
    /// draft() returns token 0 with probability 0 for the rest.  Before prepare() / the first draft(); a later change drops the captured graphs.
    void set_max_drafts(int n);

    uint64_t vram_bytes() const { return vram_; }
    int64_t n_draft_vocab() const { return n_dvocab_; }
    int n_embd() const { return N_; }
    int mask_token() const { return mask_token_; }   ///< the token whose embedding fills the block's 7 mask rows (the split engine stages the block's embeddings itself)
    const std::vector<int>& tap_layers() const { return taps_; }
    cudaStream_t stream() const { return cs_; }
    /// Diagnostics (STRATA_DF2_DUMP): the final normalized block rows of the last draft(), host copy.
    bool debug_rows(std::vector<float>& rows, std::string& err);
    /// Tests: the whole ring as it is now (layer-major: per layer K then V, each [8 heads][kRing slots][128]) and its slot tags (epoch << 32 | position, -1 none).
    bool debug_ring(std::vector<float>& kv, std::vector<int64_t>& tags, std::string& err);

    double ms_draft = 0, ms_gpu = 0, ms_ctx = 0, ms_capture = 0;
    int64_t rounds = 0;

private:
    struct Tensor { std::string name, dtype; int64_t rows = 0, cols = 0; uint64_t off = 0, bytes = 0; };
    const Tensor& find(const std::string& name) const;
    const void* q8(const std::string& name) const { return blob_ + find(name).off; }
    const float* f32(const std::string& name) const { return reinterpret_cast<const float*>(blob_ + find(name).off); }
    /// planar Q8 matrices (1..3 of them, same input width) times the planar activations in xqp_ / xdp_ (ncols columns)
    void gemm(std::initializer_list<const char*> names, std::initializer_list<float*> outs, int ncols, cudaStream_t cs, const uint8_t* xq, const float* xd);
    bool record_context(const float* taps, int n, cudaStream_t cs, std::string& err, bool rope = true);
    bool record_block(cudaStream_t cs, std::string& err, bool rope = true);
    bool capture(int n, const float* taps, std::string& err);
    bool alloc(std::string& err);
    struct Prof;
    Prof* prof_ = nullptr;
    void stamp(const char* name);
    void prof_collect();
    void prof_report() const;
    void prof_free();

    int device_ = -1;
    cudaStream_t cs_ = nullptr;
    std::vector<Tensor> tensors_;
    uint8_t* blob_ = nullptr;
    bool blob_shared_ = false;      ///< blob_ (and the bound head) belong to the parent drafter (clone_for_slot)
    float* landing_ = nullptr;
    float* dpend_ = nullptr;        ///< pinned host: deferred taps, row = position % kRing2048
    int64_t dpend_lo_ = 0, dpend_hi_ = 0;
    bool make_streams(std::string& err);
    uint64_t vram_ = 0;
    bool q4_ = false;   // the pack's matrices are Q4_0 (tools/dflash2_pack.py --q4_0)
    int N_ = 2560, L_ = 5, NH_ = 32, NKV_ = 8, HD_ = 128, I_ = 7680, window_ = 2048, group_ = 16, rank_ = 256, mask_token_ = 0;
    float theta_ = 10000.0f, eps_ = 1e-6f;
    std::vector<int> taps_;
    int64_t vocab_ = 0;

    // target pieces
    EmbedFn embed_;
    const void* head_w_ = nullptr;
    int head_type_ = -1;
    int64_t n_dvocab_ = 0;
    int32_t* dvocab_ = nullptr;
    uint8_t* dhead_ = nullptr;      ///< the draft vocabulary's head rows (owned), or null with head_w_ the whole head
    SharedHead shared_;             ///< offered by the MTP drafter (DF-05)
    bool dhead_shared_ = false;     ///< dhead_ / dvocab_ are the MTP drafter's, not freed here
    bool bound_ = false;

    // device buffers
    int32_t* d_meta_ = nullptr;     ///< [0..7] context positions, [8..15] block positions, [16] anchor token, [17] anchor position, [24..31] block tokens
    int32_t* h_meta_ = nullptr;     ///< host staging of d_meta_ (mapped; m_meta_ is its device alias)
    int32_t* m_meta_ = nullptr;
    int32_t* h_out_ = nullptr;      ///< pinned: 7 draft tokens then 7 probabilities
    float* hp_out_ = nullptr;
    float *kring_[8] = {}, *vring_[8] = {};
    float *rt_ctx_ = nullptr, *rt_blk_ = nullptr;   ///< rope tables of the context rows and of the block rows: one allocation, rt_blk_ = rt_ctx_ + 8 * 128
    double* rope_inv_ = nullptr;   ///< theta^(-2 i / 128), once
    cudaEvent_t ev_hs_[2] = {};     ///< the selector's projection runs beside the head
    uint8_t* xq_ = nullptr;         ///< the head's q8_1 activations
    uint8_t* xil_ = nullptr;        ///< their interleaved copy (the head's native_mmvq_il kernel, CUDA)
    uint8_t* xqp_ = nullptr;        ///< planar activations of the drafter's own GEMMs (kernels/dflash2.hpp)
    float* xdp_ = nullptr;
    uint8_t* xqc_ = nullptr;        ///< the same for the context update, which runs beside the block
    float* xdc_ = nullptr;
    cudaStream_t cs2_ = nullptr;    ///< the context update's branch of the round graph
    cudaEvent_t ev_fork_ = nullptr, ev_join_ = nullptr, ev_ctx_[8] = {};
    bool fork_ = false;             ///< recording the forked graph: layer l's attention waits for ev_ctx_[l]
    float *fcy_ = nullptr, *hctx_ = nullptr, *kc_ = nullptr, *vc_ = nullptr;
    float *h_ = nullptr, *xn_ = nullptr, *xc_ = nullptr, *dyn_ = nullptr, *q_ = nullptr, *qn_ = nullptr, *k_ = nullptr, *v_ = nullptr, *kblk_ = nullptr;
    float *ao_ = nullptr, *o_ = nullptr, *gate_ = nullptr, *up_ = nullptr, *mlp_ = nullptr, *part_ = nullptr, *fn_ = nullptr;
    float *logits_ = nullptr, *topval_ = nullptr, *hsel_ = nullptr, *out_prob_ = nullptr;
    int32_t *toptok_ = nullptr, *out_tok_ = nullptr;
    uint32_t* topscratch_ = nullptr;
    cudaGraphExec_t exec_[kRows + 1] = {};
    const float* graph_taps_ = nullptr;
    int ndraft_ = kDrafts;          ///< drafts a window can use (set_max_drafts)
    int64_t have_ctx_ = 0;
    // ---- ring log (see begin_request)
    static constexpr int kLogChunk = 256;   ///< positions per log chunk (allocated on first use: a window logged near the end of a long prompt costs 8 chunks)
    struct LogChunk { std::vector<int8_t> q; std::vector<float> d; std::vector<uint8_t> ok; };   ///< codes [256][2560], scales [256][80], row logged
    struct Lineage {
        std::vector<int32_t> hist;                        ///< the token of every position the ring was filled for (serve)
        std::vector<std::shared_ptr<LogChunk>> chunks;    ///< the activation rows by position / 256 (copy on write: a stashed copy shares them)
        bool logged(int64_t pos) const;
        uint64_t bytes() const;
        void truncate(int64_t keep);                      ///< forget the rows of positions >= keep
        LogChunk& writable(size_t chunk);                 ///< the chunk, allocated or un-shared first
    };
    Lineage cur_;                                         ///< the lineage the ring belongs to
    std::deque<Lineage> stash_;                           ///< parked lineages, newest last
    bool log_on_ = false;           ///< set by init_ring_log() once the pinned staging exists
    uint64_t stash_budget_ = 256ull << 20;
    int8_t* h_logq_ = nullptr;      ///< pinned: the last update's codes [8][2560] ...
    float* h_logd_ = nullptr;       ///< ... and scales [8][80], filled by the update's graph
    int64_t pend_pos0_ = -1;        ///< positions the update in flight will leave in h_log*
    int pend_n_ = 0;
    int64_t slot_pos_[kRing] = {};  ///< host shadow of the device ring tags: (epoch << 32 | position) of the row each slot holds, -1 none
    uint8_t* rb_q_ = nullptr;       ///< rebuild scratch (device): the window's codes, scales and positions
    float* rb_d_ = nullptr;
    int32_t* rb_pos_ = nullptr;
    uint64_t rb_bytes_ = 0;
    void log_stage(int64_t pos0, int n);       ///< an update of rows [pos0, pos0 + n) is queued: remember where its h_log* rows go, mark the slots
    void log_flush();                          ///< after the stream sync: move the staged rows into the lineage's log
    bool log_row(int64_t pos, const int8_t** q, const float** d) const;
    bool rebuild(int64_t lo, int64_t hi, int64_t* rebuilt, int64_t* missing, std::string& err);
    void stash_push(Lineage&& l);
    void invalidate_ring();                    ///< a new epoch: every slot is stale, the shadow forgets them
    bool record_kv(int n, cudaStream_t cs, const uint8_t* xq, const float* xd, const int32_t* pos, std::string& err);
    int64_t ring_hi_ = 0;           ///< one past the highest position written in this epoch
    bool ctx_pending_ = false;      ///< context_async() queued rows nobody has waited for yet
    int32_t epoch_ = 0;             ///< stamped into the ring tags: a new epoch makes every older slot invalid
    int64_t* tag_ = nullptr;        ///< per ring slot: (epoch << 32 | position) of the row it holds, -1 none
    cudaEvent_t ev_a_ = nullptr, ev_b_ = nullptr;
    bool timing_ = false;
    bool launched_ = false;         ///< draft_launch ran, draft_finish has not
    int launch_n_ = 0;
    int64_t launch_pos0_ = 0;
    std::chrono::steady_clock::time_point launch_t0_{};
};

}  // namespace strata::core
