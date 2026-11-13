// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/core/verify.hpp - plan v0.3 P6: the speculative VERIFY window.
//
// T tokens at consecutive positions p0 .. p0+T-1 - the last accepted token and T-1 drafts - go through all 48
// layers in ONE captured graph, and the head's argmax is produced for every one of them.  Token t's argmax is
// what plain greedy decode would produce after token t, BIT FOR BIT: every kernel here is either the single-token
// kernel applied per token, or a multi-token kernel whose per-token arithmetic is the single-token kernel's
// (multi-column MMVQ in exact mode, the T-token GDN kernels, the per-token hit activation, the multi-token CPU
// expert rows).  So a draft is accepted exactly when greedy decode would have produced it.
//
// What the window costs is the dense weights read ONCE for T tokens and the union of the T tokens' missed
// experts on the CPU (measured on decode traces: 1.75x one token's misses for T=2, 2.4x for 3, 3.05x for 4).
//
// STATE.  The window appends K/V and indexer keys for all T positions and leaves the GDN state untouched.
// `commit(n_keep)` then makes the first `n_keep` tokens permanent: the GDN conv history and recurrent state are
// advanced by replaying those tokens from inputs the window stored, the indexer's key tail is restored from a
// snapshot and the accepted keys re-appended (a rejected key can land in a slot the current block still needs),
// and the PLE history is set to its snapshot after token n_keep-1.  K/V cells past the accepted prefix are simply
// overwritten when those positions are processed again, before any query can read them.
//
// Requires the default native decode configuration (native projections, fused GR, fused GDN, fast attention and
// selection, native indexer) and a profile-filled VRAM expert tier with its residency table on the device.
#pragma once

#include <cstdio>

#include "strata/core/expert_source.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/moe_window.hpp"
#include "strata/core/session.hpp"
#include "strata/kernels/sampler.hpp"

#include "strata/core/gpu.hpp"

#include <cstdint>
#include <chrono>
#include <map>
#include <string>
#include <vector>

namespace strata::core {

class NativeHead;

/// The CPU pool for a window: x_f (n_tok, n_embd), ids (n_tok, k) -> out (n_tok * k, n_embd), hit rows zeroed.
using PoolMultiFn = MoePoolFn;

struct VerifyHits {
    const int32_t* d_res = nullptr;      ///< device [n_layers * n_expert] slot or -1
    const uint8_t* cache_base = nullptr; ///< slot 0 of the VRAM expert arena
    const uint64_t* slot_off = nullptr;   ///< E-6: host per-slot offsets when slots differ in size (null: slot * blob)
    int64_t n_slots = 0;                  ///< E-6: how many (for the device copy)
    int64_t blob = 0;
    const int32_t* h_res = nullptr;       ///< host [n_layers * n_expert] slot or -1: a window whose layers have every
                                          ///< expert resident runs without the host (null: never)
};

class Verifier;

/// LAYER SPLIT: what a stage asks of the stage after it.  Everything that crosses a stage boundary of the windows
/// goes through here: the split point's rows, the requests' control (windows, commits, sampling settings) and the
/// results coming back from the last stage (its logits), so a next stage that shares nothing else with this one
/// (another process, another machine) can stand behind it.  The calls marked "same process" hand a pointer over
/// instead of its contents; only a stage in this process can serve them.  The engine's other paths that reach across
/// stages (a CPU pool, a RAM expert region and a residency table shared by every stage, the host driving each stage's
/// verifiers and sessions) are marked "LAYER SPLIT, same process only" where they are made.
class StageLink {
public:
    virtual ~StageLink() = default;
    /// Rows [row0, row0 + n) of this stage's hand-off (`Verifier::handoff_floats` each) are written: the next stage's.
    virtual void rows(const float* hand_out, int row0, int n) = 0;
    /// The window and the batch window, continued on the next stage.  `pool` is this process's CPU pool function
    /// (a stage in another process serves its layers' experts with its own).
    virtual bool run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, int32_t* out, std::string& err) = 0;
    virtual bool run_slot_rows(const int* rows, int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool,
                               int32_t* out, std::string& err) = 0;
    virtual bool commit(int n_keep, std::string& err) = 0;
    virtual bool commit_slot_prefixes(const int* keep, std::string& err) = 0;
    virtual bool wait_commit(std::string& err) = 0;
    virtual void set_sampling(const strata::kernels::SamplerParams& sp) = 0;
    virtual void set_slot_sampling(int slot, const strata::kernels::SamplerParams& sp) = 0;
    /// A slot's reply tail for its penalty window and its own logit bias (same process: the pointer is copied).
    virtual void set_slot_history(int slot, const int32_t* tail, int64_t n) = 0;
    virtual bool set_slot_logit_bias(int slot, const std::vector<float>& bias, std::string& err) = 0;
    virtual void set_head_sampling(bool on) = 0;
    virtual bool set_logit_bias(const std::vector<float>& bias, std::string& err) = 0;
    /// The grammar's allowed tokens for the next window's T rows (host bitmask rows), uploaded on the last stage's
    /// device; null clears them.
    virtual bool set_allowed(const uint32_t* rows, int T, std::string& err) = 0;
    /// STRATA_ROUTE_PRIOR on or off for the next windows on every stage (off: the router's exact routing).
    virtual bool set_route_prior(bool on, std::string& err) = 0;
    /// Same process: the penalty rows the last stage's head reads, a buffer the engine fills before every window.
    virtual void set_history(const int32_t* history, int history_len) = 0;
    virtual bool read_logits(int rows, float* host, std::string& err) const = 0;
    virtual bool copy_logits(int t, float* host) const = 0;
    /// The last stage's check of its head logits (#879): the first row that held a NaN or an infinity, its position,
    /// and whether row t did.
    virtual int nonfinite_row() const = 0;
    virtual int64_t nonfinite_pos() const = 0;
    virtual bool nonfinite_in(int t) const = 0;
    /// Same process: the measurement hook writes its lines to `out` (STRATA_LOGPOS).
    virtual bool window_logprobs(const int32_t* targets, int T, int64_t pos0, int32_t extra_id, std::FILE* out,
                                 std::string& err) = 0;
    /// Same process: the last stage's final rows on its device (the drafter's input).
    virtual const float* final_R_all() const = 0;
};

/// The next stage in this process: the calls go to its Verifier, and the rows are copied into its hand-off when it
/// has one of its own (a stage on another GPU, whose context cannot read this one's host memory).
class InProcessStage final : public StageLink {
public:
    void bind(Verifier* next, void* pool_user) { next_ = next; user_ = pool_user; }
    void rows(const float* hand_out, int row0, int n) override;
    bool run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, int32_t* out, std::string& err) override;
    bool run_slot_rows(const int* rows, int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool,
                       int32_t* out, std::string& err) override;
    bool commit(int n_keep, std::string& err) override;
    bool commit_slot_prefixes(const int* keep, std::string& err) override;
    bool wait_commit(std::string& err) override;
    void set_sampling(const strata::kernels::SamplerParams& sp) override;
    void set_slot_sampling(int slot, const strata::kernels::SamplerParams& sp) override;
    void set_slot_history(int slot, const int32_t* tail, int64_t n) override;
    bool set_slot_logit_bias(int slot, const std::vector<float>& bias, std::string& err) override;
    void set_head_sampling(bool on) override;
    bool set_logit_bias(const std::vector<float>& bias, std::string& err) override;
    bool set_allowed(const uint32_t* rows, int T, std::string& err) override;
    bool set_route_prior(bool on, std::string& err) override;
    void set_history(const int32_t* history, int history_len) override;
    bool read_logits(int rows, float* host, std::string& err) const override;
    bool copy_logits(int t, float* host) const override;
    int nonfinite_row() const override;
    int64_t nonfinite_pos() const override;
    bool nonfinite_in(int t) const override;
    bool window_logprobs(const int32_t* targets, int T, int64_t pos0, int32_t extra_id, std::FILE* out,
                         std::string& err) override;
    const float* final_R_all() const override;

private:
    Verifier* next_ = nullptr;
    void* user_ = nullptr;   ///< what the next stage's pool calls get
};

class Verifier {
public:
    Verifier() = default;
    ~Verifier();
    Verifier(const Verifier&) = delete;
    Verifier& operator=(const Verifier&) = delete;

    /// The watchdog's view of the window in flight (issue #31): the layer, the GPU's sequence, the flags.
    void diag(std::FILE* f) const;

    /// The device's measured kernel forms for a window (the GDN step's, the multi-row BF16 GEMV's for the window's
    /// shapes), each measured once per device; `init` asks for them too.  Called before the first prompt, so their
    /// measurement (about 1 s on an RTX 4070) is not in its time to the first token.  Throws on a device error.
    static void tune_device(const ModelGeometry& g, void* stream);

    /// `max_t` <= kVerifyMaxT.  `head` may be null (the canonical head is then run per token).
    bool init(const WeightTable& wt, const ModelGeometry& g, SessionState& ss, const VerifyHits& hits,
              const NativeHead* head, int max_t, std::string& err);

    /// One window: `tokens[0..T)` at positions pos0.., the pool served per layer; `out[t]` = argmax after token t.
    /// The PLE rows are gathered here from `ss.ple_prev` and the tokens.  Captures the T-token graph on first use.
    bool run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool, void* user, int32_t* out, std::string& err);
    /// Whether the last run() captured its window's graph: that round's time is mostly the capture's, not the window's
    bool captured_last() const { return captured_last_; }
    /// Whether a window of T tokens can run without running the card out of memory: its graph is captured already,
    /// or the VRAM free leaves room to capture it (half as much again as the most a capture has taken, and 64 MiB)
    bool window_fits(int T) const;
    /// The head's raw logits for the first `rows` tokens of the last window (the last stage's head):
    /// `rows` x n_vocab floats into `host`.  For --dump-logits; call after `run`, before the next window.
    bool read_logits(int rows, float* host, std::string& err) const;
    /// The sampling the verify window's head applies (temperature / top_p / top_k / seed).  Set per
    /// request; greedy by default.  The sampling itself runs OUTSIDE the captured graph - its
    /// parameters would otherwise be baked forever - so this can change between requests freely.
    void set_sampling(const strata::kernels::SamplerParams& sp) {
        sampling_ = sp;   // row t of a window at pos0 draws Philox(seed, pos0 + t): see run()
        sampling_.logit_bias = logit_bias_host_.empty() ? nullptr : logit_bias_device_;
        sampling_.allowed = allowed_on_ ? allowed_device_ : nullptr;
        if (next_) next_->set_sampling(sp);
    }
    // One request's fixed vocabulary bias. Empty clears it; copied only when changed, on the final stage's device.
    bool set_logit_bias(const std::vector<float>& bias, std::string& err);
    // Grammar-constrained decoding (src/core/grammar.cpp): row t's allowed tokens for the next window, T rows of
    // (n_vocab + 31) / 32 words from the host, copied before every window to the final stage's device.  Null clears
    // them; set, the head samples outside the captured graph as for a logit_bias.
    bool set_allowed(const uint32_t* rows, int T, std::string& err);
    // STRATA_ROUTE_PRIOR for the next windows (a request's route_prior=0|1): off is the router's exact routing.  The kernel
    // reads lambda from the device, so the captured windows stay as they are.  A no-op without STRATA_ROUTE_PRIOR.
    bool set_route_prior(bool on, std::string& err);

    /// The penalty histories for `sampling_.penalty_last_n`: ONE ROW PER WINDOW ROW, T rows of `history_len`
    /// int32 slots at that stride (`strata::kernels::penalty_rows` builds them), most recent token LAST, unused
    /// front slots -1 (the kernel reads only the tail window).  Row t follows the window's drafts 1..t - staging
    /// row 0 alone (before 0.1.19) left the drafted rows with unwritten histories.  Null disables the penalties
    /// entirely - the neutral run's sampling call is byte-for-byte what it was.  The engine re-uploads the rows
    /// before every window; the buffer must hold kVerifyMaxT rows and stay alive across the request.
    void set_history(const int32_t* history, int history_len) {
        hist_d_ = history;
        hist_len_ = history_len;
        if (next_) next_->set_history(history, history_len);
    }
    /// Off: `run` skips the request's head sampling and `out` is the recorded greedy pick.  For windows whose
    /// picks are discarded - a prompt read through windows commits every token - so they cost no sampler launch
    /// or sync and never read a history staged for another position.
    void set_head_sampling(bool on) { head_sampling_ = on; if (next_) next_->set_head_sampling(on); }

    /// LAYER SPLIT (multi-GPU): this verifier runs layers [layer_begin, layer_end) of every window.  A stage that
    /// does not start at layer 0 takes its residual from `handoff_in` instead of embedding the tokens; a stage that
    /// does not end at the last layer writes its residual to `handoff_out` and has no head.  The hand-off holds,
    /// per token, the residual R (hc x n_embd), the last layer's pending write bo (n_embd) and inject (hc): the next
    /// stage folds that write into its first read exactly as the unsplit window does, so the split is bit-exact.
    /// Both pointers are host USM of this stage's GPU, which its kernels read and write.  Stages on one GPU share
    /// the buffer between them; when the next stage is on another GPU, its `handoff_in` is a buffer of its own and
    /// `run` copies the rows into it.
    /// Set before `init`.  Default: the whole model, no hand-off.
    void set_stage(int64_t layer_begin, int64_t layer_end, const float* handoff_in, float* handoff_out) {
        lb_ = layer_begin; le_ = layer_end; hand_in_ = handoff_in; hand_out_ = handoff_out;
    }
    /// The DFlash2 drafter's taps: the completed output (the mean of the hyper-connection streams) of these layers,
    /// gathered for every window row into tap_dev() ([T][ntap][n_embd]).  Only the layers this stage runs below the
    /// last are kept.  Set before init; empty (the default) allocates nothing and records nothing.
    void set_taps(const std::vector<int>& layers) { tap_req_ = layers; }
    /// The taps off after init (the start's measurement released DFlash2): the window graphs, captured with the tap
    /// kernels, are dropped and captured again without them on their next use.
    void taps_off();
    const float* tap_dev() const { return tap_dev_; }
    int n_taps() const { return (int) tap_layers_.size(); }

    /// The rows the hand-off buffers of `set_stage` hold (0: kVerifyMaxT, the default).  STRATA_PIPED_MTP's groups write
    /// their slots' token and proposal rows at hbase = group * rows-per-group, past kVerifyMaxT with enough slots.
    void set_handoff_rows(int rows) { handoff_rows_ = rows; }
    /// The next stage: `run` and `commit` continue into it (its pool calls get `next_user`); sampling settings
    /// and `final_R` are the last stage's.  Everything goes through a StageLink (an InProcessStage here).
    void set_next(Verifier* next, void* next_user) {
        local_next_.bind(next, next_user);
        next_ = next != nullptr ? &local_next_ : nullptr;
    }
    /// floats per token in a hand-off buffer
    static int64_t handoff_floats(const ModelGeometry& g) { return (int64_t) g.hc * g.n_embd + g.n_embd + g.hc; }
    /// The previous stage in this process wrote rows [row0, row0 + n) of its hand-off `from`: they become this stage's
    /// (copied into its own hand-off when it has one: a stage on another GPU).  InProcessStage::rows.
    void accept_rows(const float* from, int row0, int n);

    /// Keep the first `n_keep` (1..T) tokens of the last window; advances `ss.ple_prev` by them.
    bool commit(int n_keep, std::string& err);
    /// The adaptive tier's routing counts: usage[l * n_expert + e] += 1 for each expert e routed at layer l (of this
    /// verifier's layers) by the last window's first `rows` rows, the rows whose input was the text's (the last
    /// accepted token and the accepted drafts).  The rejected rows' routing is not counted, and neither is a row
    /// counted twice: the counts follow the text, not how it was cut into windows (window-invariant output, #152).
    void add_routing(float* usage, int64_t n_expert, int rows) const;
    /// Record every layer's routed ids for add_routing (off: no copies).  Before the first window is captured.
    void set_route_record(bool on) { route_rec_ = on; }
    /// commit() returns without waiting for its graph (a single-GPU session sets it): the next window follows it on
    /// the same queue and the drafter reads nothing it writes, so it overlaps the draft.  Whoever reads or writes the
    /// session from another queue or the host afterwards (a new request, a checkpoint, the prompt path, the end of a
    /// run) calls wait_commit() first (upstream 42b4299, a454dbb).  STRATA_COMMIT_SYNC=1 keeps the wait.
    static void set_commit_async(bool on);
    /// Waits for the last commit graph when commit() did not (an event recorded after it, not the whole device);
    /// false with `err` when it failed.  Free when nothing is pending.
    bool wait_commit(std::string& err);

    // ---- PIPELINED WINDOWS (--pipeline-windows, a layer split on two GPUs; upstream bfa532e9).  One conversation's
    // windows with the stages overlapped: stage 0 runs window K+1 while stage 1 still runs window K.  The same window
    // as `run`, driven without blocking the host, so one host thread keeps a window in flight on each stage.  Two
    // verifiers per stage (one per window parity) share the stage's stream and each has its own hand-off.
    // `pl_launch` stages and launches (it never captures: `capture_all` first, with nothing in flight), `service`
    // serves the layers whose doorbells have rung and returns at once, `done` polls the window's completion,
    // `pl_finish` reads the picks, `pl_commit_async` queues the commit.  Nothing chains to `next_`: the caller drives
    // every stage.
    /// The compute stream to use instead of a private one (the two verifiers of one stage share it).  Before `init`.
    void set_stream(strata::gpu::Stream s) { ext_stream_ = s; }
    /// Turns the device-planned layers and the all-resident window graph off: with windows in flight the adaptive
    /// tier marks an expert evicted on the host before the device's residency table follows.  Before `init`.
    void set_always_publish(bool on) { always_publish_ = on; }
    strata::gpu::Stream stream() const { return cs_; }
    int device() const { return device_; }
    /// Capture every window size and the commit graph now (a capture syncs the stream: never with a window in flight).
    bool capture_all(std::string& err);
    bool pl_launch(int T, const int32_t* tokens, int64_t pos0, std::string& err);
    /// Stage a window ahead of its launch (positions, and the PLE rows from `ple_prev` = the two tokens before the
    /// window as they WILL be).  A later `pl_launch` of the same window (T, pos0, tokens, and `ss.ple_prev` equal to
    /// `ple_prev` by then) skips the staging; anything else stages again.
    bool prestage(int T, const int32_t* tokens, int64_t pos0, const int32_t ple_prev[2], std::string& err);
    /// 1: every layer served; 0: the GPU has not reached the next layer yet; -1: an error (`err`).
    int service(PoolMultiFn pool, void* user, std::string& err);
    bool in_flight() const { return fl_active_; }
    /// The window's graph completed; false while it runs.  An error sets `err`.
    bool done(std::string& err);
    /// After `done`: the profile, the last stage's host sampling and picks (`out` may be null on an earlier stage).
    bool pl_finish(int32_t* out, std::string& err);
    /// The commit without a host sync; `ss.ple_prev` advances now (host side).  A second call for the same window
    /// (after its state was restored) replays it with another count.
    bool pl_commit_async(int n_keep, std::string& err);
    /// Fold another verifier's counters and GPU profile into this one's (the two verifiers of one stage report once).
    void absorb_stats(Verifier& o);
    /// The watchdog's line for a pipelined verifier: in flight, layers served, the GPU's ring and flags, its events.
    void diag_pipelined(std::FILE* f, const char* name) const;

    // ================================ SEVERAL SEQUENCES IN ONE WINDOW (upstream PR #559) ================================
    //
    // A batch window holds S INDEPENDENT sequences, one token each: row s is slot s, at slot s's own position,
    // reading and writing slot s's own state (GDN recurrence and conv history, QSA K/V and indexer, PLE history),
    // which lives in `slots[s]` - a session carved like this verifier's own (same max_cells).  Everything that is per
    // row already (hyper-connections, dense projections, router, shared expert, the routed experts on the GPU and the
    // CPU, the head) runs once over the S rows, so the weights are read once per window for all the sequences.  Row
    // s's arithmetic is the single-token window's, so a slot's greedy tokens are its solo greedy tokens (modulo the
    // multi-token CPU kernel choice: STRATA_IQ_MT_MIN=1).  No drafts (MTP) in a batch window.  `init_slots` once
    // after `init` (S <= max_t); a GPU that runs windows as segments (doorbell_visible false) takes none.
    bool init_slots(const std::vector<SessionState*>& slots, std::string& err);
    int n_slots() const { return (int) slots_.size(); }
    /// One batch window over slots [0, S): tokens[s] at positions pos[s]; out[s] = the pick after it.
    bool run_slots(int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool, void* user, int32_t* out,
                   std::string& err);
    /// The same over the S slots `rows` (row t is slot rows[t], any distinct slots in any order): the slots not listed
    /// are not touched, so an idle slot keeps its state (a finished conversation it may continue later; #465).
    bool run_slot_rows(const int* rows, int S, const int32_t* tokens, const int64_t* pos, PoolMultiFn pool, void* user,
                       int32_t* out, std::string& err);
    /// Keep every row of the last batch window: each slot's state advances by its one token.
    bool commit_slots(std::string& err);
    /// Commit an accepted prefix in each contiguous slot group of the last window (--batch-mtp, upstream 8cfb3fd7):
    /// `keep` has one entry per slot (indexed by slot), each in 1..that slot's group length.
    bool commit_slot_prefixes(const int* keep, std::string& err);
    /// --batch-mtp: slot rotation makes many row layouts, so bound the captured batch graph pairs (LRU).  0 = unbounded.
    void set_batch_graph_limit(size_t n) { batch_graph_limit_ = n; }

    // ---- The stages of a layer split as a PIPELINE (upstream PR #559, --batch-groups).  A batch window over the slot
    // GROUP [base, base + S) is launched on ONE stage with its commit right behind it on the stage's queue (a batch
    // window keeps every row, so the commit needs no host decision), and the host serves the rings of every stage
    // that has a window in flight from one thread (batch_poll does not block).  Stage k can then run group g while
    // stage k+1 runs group g-1.  Rows of group `base` use hand-off rows [base, base + S), so groups never share one;
    // a stage on another GPU gets its group's rows copied over when the window is done.
    bool batch_launch(int base, int S, const int32_t* tokens, const int64_t* pos, std::string& err);
    /// 1 = this stage's window and commit are done (the last stage's picks are in batch_out), 0 = still running,
    /// -1 = an error (err).  Serves every layer that has rung so far.
    int batch_poll(PoolMultiFn pool, void* user, std::string& err);
    bool batch_busy() const { return b_running_; }
    const int32_t* batch_out() const { return b_out_; }
    // ---- STRATA_PIPED_MTP (upstream 17878ac57): the pipelined groups with --batch-mtp's drafts.  A group's window holds
    // each of its slots' token and proposals, so how many rows to keep is known only once the LAST stage has picked -
    // and by then this stage may run another group's window, which reuses the scratch the commit replays from.  So
    // right behind the window a save graph copies those rows (each GDN layer's conv and recurrence inputs, each QSA
    // layer's raw indexer keys, the PLE history snapshot) to the group's own rows [hbase, hbase + S) of the saved
    // buffers, and the commit, queued after the verdict, replays from there.  Everything else the commit reads is per
    // slot already.
    /// The saved buffers for `rows` hand-off rows (after init_slots).
    bool init_deferred_commit(int rows, std::string& err);
    bool deferred_commit_on() const { return dc_rows_ > 0; }
    /// The window over the slots `rows` (a slot's rows contiguous) at hand-off rows [hbase, hbase + S), its save right
    /// behind it, and no commit (batch_commit_rows, after the verdict).  The PLE history advances at the commit.
    bool batch_launch_rows(const int* rows, int S, int hbase, const int32_t* tokens, const int64_t* pos,
                           std::string& err);
    /// The commit of that window's accepted prefixes, `keep` per slot ID (1..the slot's rows), in two halves: the host's
    /// (the accepted prefixes staged, the PLE hash's last two tokens) now, and `batch_commit_rows_submit` queueing the
    /// graph on this stage's stream, with no host wait.  The caller submits it only while no window waits on the host
    /// on this card (a graph queued behind such a window did not return on the B70), and before the slots' next window
    /// on this stage.  `rows`, `S`, `hbase`, `tokens` and `pos` are the window's, as batch_launch_rows got them.
    bool batch_commit_rows(const int* rows, int S, int hbase, const int* keep, const int32_t* tokens,
                           const int64_t* pos, std::string& err);
    bool batch_commit_rows_submit(const int* rows, int S, int hbase, std::string& err);
    bool last_stage() const { return g_ != nullptr && le_ == g_->n_layers; }
    /// A slot's sampling (temperature / top_p / top_k / min_p / seed, and with set_slot_history and
    /// set_slot_logit_bias its penalties and bias): its row is drawn again with Philox(seed, position), as a solo
    /// window draws it.  Greedy by default.  Set on the first stage of a layer split, it reaches the last (which
    /// draws).
    void set_slot_sampling(int slot, const strata::kernels::SamplerParams& sp) {
        if (slot >= 0 && slot < (int) slot_sp_.size()) {
            slot_sp_[(size_t) slot] = sp;
            // a slot's own bias array stays its row's (set_slot_logit_bias; the request's sp carries no pointer)
            slot_sp_[(size_t) slot].logit_bias =
                slot < (int) slot_bias_dev_.size() ? slot_bias_dev_[(size_t) slot] : nullptr;
        }
        if (next_) next_->set_slot_sampling(slot, sp);
    }
    /// A slot's reply tail for its penalty window: the `n` tokens its session consumed since the reply began (the
    /// last penalty_last_n of them are counted), the fed-back head not in it yet.  Set before every batch window
    /// that holds the slot while it has a penalty; reaches the last stage, which draws the rows.
    void set_slot_history(int slot, const int32_t* tail, int64_t n) {
        if (slot >= 0 && slot < (int) slot_tail_.size()) {
            if (n > 0) slot_tail_[(size_t) slot].assign(tail, tail + n);
            else slot_tail_[(size_t) slot].clear();
        }
        if (next_) next_->set_slot_history(slot, tail, n);
    }
    /// A slot's logit bias (n_vocab floats, empty = none), added before its row's penalties and filters as a solo
    /// request's is.  Reaches the last stage.
    bool set_slot_logit_bias(int slot, const std::vector<float>& bias, std::string& err);

    /// Row `t` of the last window's head logits (the distribution token t of the window came from: before sampling,
    /// temperature and penalties) into `host` (vocab() floats); false when there is none.  OpenAI's logprobs.
    bool copy_logits(int t, float* host) const;
    int64_t vocab() const { return n_vocab_; }
    /// #879 GUARD (upstream f332a855): the first row of the last finished window (run, run_slot_rows, batch_poll,
    /// pl_finish) whose head logits held a NaN or an infinity, else -1; nonfinite_pos() is that row's position.  Every
    /// window graph checks its T rows on the device beside the greedy pick (logits_nonfinite_rows into mapped flags,
    /// read after the sync the window already does: no extra sync).  On the stage with the head (a layer split asks
    /// the last stage).  STRATA_NAN_GUARD=0: off (always -1).  STRATA_NAN_INJECT=<n>[,<n>...] (debug): the n-th
    /// window the guard checks in this process (counted from 1 over every stage-with-head verifier: prompt-tail
    /// windows, decode windows and batch windows alike) reports row 0 as non-finite, so the whole response can be
    /// tested.
    int nonfinite_row() const { return next_ != nullptr && !last_stage() ? next_->nonfinite_row() : nf_row_; }
    int64_t nonfinite_pos() const { return next_ != nullptr && !last_stage() ? next_->nonfinite_pos() : nf_pos_; }
    /// Row t of the last window (a batch window: each slot's own rows) held a non-finite logit.
    bool nonfinite_in(int t) const {
        if (next_ != nullptr && !last_stage()) return next_->nonfinite_in(t);
        return t >= 0 && t < 32 && ((nf_mask_ >> t) & 1u) != 0;
    }
    static bool nan_guard_on();
    /// Measurement hook (STRATA_LOGPOS, upstream 06b82fe / 5e14c19): after run(), one line per row t of the last
    /// window's head - "pos target logprob top top_logprob hit extra_logprob target_logprob_without_extra", and with
    /// STRATA_LOGPOS_TOPK=K the K most likely tokens as `id:logprob` - where row t is the distribution at pos0 + t,
    /// targets[t] the token at pos0 + t + 1, extra_logprob that of `extra_id` in the row, and the last column the
    /// target's log-probability renormalized over every token but `extra_id` (nan when they do not apply).
    bool window_logprobs(const int32_t* targets, int T, int64_t pos0, int32_t extra_id, std::FILE* out,
                         std::string& err);

    /// Token t's residual after the last layer, (hc, n_embd) on the device, valid until the next `run`.
    const float* final_R(int t) const;
    const float* final_R_all() const { return next_ ? next_->final_R_all() : R_; }

    /// The GPU plan the pool writes each layer (VRAM hits + the PCIe share of the misses); give it to the
    /// dispatch (`ExpertDispatch::plan`) before the first `run`.
    GpuPlanSink* plan_sink() { return win_.sink(); }
    /// The latent MoE (Edge0): the window's mapped D rows, their doorbell and their Q8_0 image, for the pool's
    /// CPU share (ExpertDispatch's lat_* fields); null when the pack is not a latent one.
    const float* latent_xl_mapped() const { return win_.latent_xl_mapped(); }
    const uint32_t* latent_xlseq() const { return win_.latent_xlseq(); }
    const uint8_t* latent_act0_mapped() const { return win_.latent_act0_mapped(); }
    bool latent() const { return win_.latent(); }
    /// Plan v0.3 P6: split the window into two token groups and pipeline the CPU experts of one with the GPU work
    /// of the other (default on).  Set before the first `run`.
    void set_split(bool on) { split_ = on; }
    /// Plan v0.3 P6: how the PCIe share of the misses reaches the GPU: 0 = DMA into staging (the copy engine works
    /// beside the CPU; best when the CPU is compute-bound, the i-quants), 1 = the grouped kernel reads the mapped
    /// arena directly, 2 = a copy kernel stages it inside the graph (no API calls on the pool's thread; best when
    /// the CPU is RAM-bound, Q2_0).  Set before the first `run`.
    void set_pcie_mode(int mode) { win_.sink()->pcie_mode = mode; }
    /// Whether the window reads the PCIe share beside its VRAM hits (STRATA_FETCH_BRANCH), for the split's estimate.
    bool fetch_branch() const { return fetch_branch_; }
    /// the pool never plans a PCIe share (--pcie-frac 0): the window skips that path.  Before the first run.

    double ms_wait = 0, ms_pool = 0, ms_host = 0, ms_commit = 0;
    int64_t windows = 0;
    /// STRATA_VERIFY_PROFILE=1 - GPU stage counts since the last call (millions of device-clock ticks per
    /// window), as one line; empty when off, unsupported, or no windows have been profiled.
    std::string profile_report();

private:
    bool capture(int T, std::string& err);
    strata::kernels::SamplerParams sampling_ = [] {
        strata::kernels::SamplerParams s;
        s.greedy = true;
        s.temperature = 0.0f;
        return s;
    }();   ///< greedy by default; per-request via set_sampling
    const int32_t* hist_d_ = nullptr;   ///< penalty-history row (set_history); null = no penalties apply
    int hist_len_ = 0;
    bool head_sampling_ = true;          ///< set_head_sampling
    std::vector<float> logit_bias_host_;
    float* logit_bias_device_ = nullptr;
    uint32_t* allowed_device_ = nullptr;   ///< kVerifyMaxT rows of the grammar's mask (set_allowed)
    bool allowed_on_ = false;
    float* rp_lambda_d_ = nullptr;          ///< STRATA_ROUTE_PRIOR's lambda on the device (0: off; set_route_prior)
    bool rp_on_ = true;
    unsigned long long* rp_stats_ = nullptr;   ///< STRATA_ROUTE_PRIOR: this GPU's counters
    int device_ = -1;                    ///< the device `init` ran on: run/commit switch to it (layer split)
    unsigned long long* rr_stats_ = nullptr;   ///< STRATA_ROUTE_RESIDENT: this GPU's counters (nullptr: none)
    bool device_plan_ = false;            ///< E-6: resident-only layers planned on the device (STRATA_VERIFY_DEVICE_PLAN)
    // Every expert of the verifier's layers resident (hits.h_res): the window runs as one graph that plans each
    // layer's experts on the device and never asks the host - no doorbell, flag waits, PCIe or CPU share (upstream
    // cfd3b72's zero-doorbell graph).  STRATA_VERIFY_RESIDENT_GRAPH=0: off.
    // batch windows (see init_slots)
    std::vector<SessionState*> slots_;
    bool batch_rec_ = false;               ///< record_window is capturing a batch window
    int brow_[8] = {};                     ///< ... and row t is slot brow_[t]
    bool last_batch_ = false;              ///< the last run was a batch window (set_plan_slot: one group)
    /// key: batch_key(rows, S, hand-off base) - the whole row layout (--batch-mtp repeats a slot over its rows)
    std::map<std::vector<int>, strata::gpu::Graph*> exec_bm_, commit_bm_;
    std::map<std::vector<int>, uint64_t> bm_used_;   ///< last use of each captured layout (LRU, with a graph limit)
    uint64_t bm_tick_ = 0;
    size_t batch_graph_limit_ = 0;         ///< 0: keep every captured batch graph; N: LRU-evict beyond N layouts
    int last_rows_[8] = {};                ///< the slots of the last batch window's rows
    static std::vector<int> batch_key(const int* rows, int S, int hbase) {
        std::vector<int> k;
        k.reserve((size_t) S + 1);
        k.push_back(hbase);
        for (int t = 0; t < S; ++t) k.push_back(rows[t]);
        return k;
    }
    int row_base_ = 0;                     ///< a batch window's first hand-off row (its slot group's base)
    // batch_launch / batch_poll
    bool b_running_ = false;
    int64_t b_k_ = 0, b_steps_ = 0;
    std::chrono::steady_clock::time_point b_last_;
    int32_t b_out_[8] = {};
    /// The host's half of a batch window: the checks, the graphs, the rows' staging (run_slot_rows, batch_launch).
    bool stage_batch(const int* rows, int S, int hbase, const int32_t* tokens, const int64_t* pos, std::string& err);
    std::vector<strata::kernels::SamplerParams> slot_sp_;
    std::vector<float*> slot_bias_dev_;              ///< per slot: its logit bias on the device (null: none)
    std::vector<std::vector<int32_t>> slot_tail_;    ///< per slot: its reply tail for the penalty window
    int32_t* slot_hist_d_ = nullptr;                 ///< the current batch window's penalty-history rows (device)
    size_t slot_hist_cap_ = 0;                       ///< ... and how many int32 fit (grows to the widest window)
    int32_t* h_commitb_ = nullptr; int32_t* m_commitb_ = nullptr;   // per slot [1, 0, pos, -1 ..], stride 2 + max_t
    int32_t* commitb_ = nullptr;
    float* tail_snap_b_ = nullptr;         ///< per (slot, QSA layer) indexer tail snapshot
    void* arena_b_ = nullptr;
    int64_t last_pos_b_[8] = {};
    bool capture_batch(const int* rows, int S, int hbase, std::string& err);
    bool capture_commit_batch(const int* rows, int S, int hbase, std::string& err);
    // STRATA_PIPED_MTP (init_deferred_commit): the saved rows, laid out as the window scratch (global layer index) with
    // dc_rows_ rows per layer instead of max_t_; the slots' accepted prefixes (mapped and device, 2 + max_t_ per slot)
    int dc_rows_ = 0;
    bool staging_deferred_ = false;   ///< stage_batch for batch_launch_rows: no capture_commit_batch
    float *dc_qkv_ = nullptr, *dc_h_ = nullptr, *dc_gate_ = nullptr, *dc_beta_ = nullptr, *dc_idx_ = nullptr;
    float* dc_hist_ = nullptr;
    void* dc_arena_ = nullptr;
    int32_t *h_commitd_ = nullptr, *m_commitd_ = nullptr, *commitd_ = nullptr;
    std::map<std::vector<int>, strata::gpu::Graph*> save_bm_, dcommit_bm_;
    bool capture_deferred(const int* rows, int S, int hbase, std::string& err);
    bool sample_rows(int S, std::string& err);
    bool res_graph_ = false;
    bool recording_res_ = false;
    strata::gpu::Graph* exec_res_[9] = {};
    bool all_resident() const;
    bool capture_res(int T, std::string& err);
    int64_t lb_ = 0, le_ = -1;           ///< set_stage: the layers this verifier runs (-1: to the last)
    std::vector<int> tap_req_, tap_layers_;   ///< set_taps: the DFlash2 drafter's tap layers (this stage's share)
    std::vector<int> tap_slot_;          ///< layer -> its tap index (-1: not tapped)
    float* tap_dev_ = nullptr;           ///< [max_t][ntap][n_embd], the window's taps (null: none)
    int handoff_rows_ = 0;               ///< set_handoff_rows (0: kVerifyMaxT)
    const float* hand_in_ = nullptr;
    float* hand_out_ = nullptr;
    StageLink* next_ = nullptr;          ///< set_next: the next stage, through this link only
    InProcessStage local_next_;
    bool ple_stage() const { return lb_ <= 1 && 1 < le_; }   ///< holds layer 1, where the PLE block runs
    bool capture_commit(std::string& err);
    bool record_window(int T, strata::gpu::Stream cs, std::string& err);
    static constexpr int kProfPer = 33;              // stamps per layer (32 left the hc-read second half's up-stamp
                                                     // at slot 32 = the next layer's slot 0: upstream e5b47dd)
    bool prof_on_ = false;
    void accumulate_profile(const unsigned long long* stamps);   ///< one window's stamps (host copy) into prof_sum_
    // pipelined windows (pl_launch ...)
    strata::gpu::Stream ext_stream_ = nullptr;   ///< set_stream: the stage's shared stream (not destroyed here)
    bool always_publish_ = false;
    bool pl_stage(int T, const int32_t* tokens, int64_t pos0, const int32_t ple_prev[2], std::string& err);
    strata::gpu::Event *ev_done_ = nullptr, *ev_commit_ = nullptr;
    unsigned long long* prof_pin_ = nullptr;   ///< host copy of the stamps (pipelined windows)
    bool fl_active_ = false, fl_prof_ = false, commit_live_ = false, pl_prestaged_ = false;
    int fl_T_ = 0;
    int64_t fl_k_ = 0, fl_total_ = 0;
    double fl_since_ms_ = 0, fl_flush_ms_ = 0, fl_launch_ms_ = 0;
    int32_t pl_prev_[2] = {-1, -1};
    unsigned long long* prof_ = nullptr;              // device: n_layers * kProfPer + 4 stamps
    std::vector<unsigned long long> prof_h_;
    double prof_sum_[2][kProfPer] = {};   // [GDN / QSA layers][stage]
    int64_t prof_windows_ = 0;

    const WeightTable* wt_ = nullptr;
    const ModelGeometry* g_ = nullptr;
    SessionState* ss_ = nullptr;
    VerifyHits hits_;
    const NativeHead* head_ = nullptr;
    int max_t_ = 0;
    int last_t_ = 0;
    int64_t last_pos0_ = 0;
    int32_t last_tokens_[8] = {};
    int64_t n_vocab_ = 0;
    strata::gpu::Stream cs_ = nullptr;
    strata::gpu::Stream df_side_[2] = {};   ///< STRATA_DF_BRANCH: the mixer's side branches of the window graph
    bool df_branch_ = false;
    strata::gpu::Stream fetch_side_ = nullptr;   ///< STRATA_FETCH_BRANCH: the PCIe share's fetch beside the VRAM hits
    bool fetch_branch_ = false;
    strata::gpu::Graph* exec_[9] = {};
    bool captured_last_ = false;
    size_t graph_bytes_max_ = 0;   ///< the most VRAM a window's capture and first run have taken
    size_t graph_free0_ = 0;       ///< the VRAM free before the capture of the window run() is running, else 0
    strata::gpu::Graph* commit_exec_ = nullptr;
    std::vector<strata::gpu::Graph*> segs_[9];
    std::vector<strata::gpu::Graph*>* seg_out_ = nullptr;   // the segments of the window being captured

    // mapped staging (host pointer, device alias)
    int32_t* h_tok_ = nullptr;   int32_t* m_tok_ = nullptr;     // T
    int32_t* h_step_ = nullptr;  int32_t* m_step_ = nullptr;    // T * kStepCount
    int32_t* h_pos_ = nullptr;   int32_t* m_pos_ = nullptr;     // T * n_head
    int32_t* h_commit_ = nullptr; int32_t* m_commit_ = nullptr; // [n_keep, n_keep-1, pos_0 .. pos_{T-1}]
    float* h_ple_ = nullptr;     float* m_ple_ = nullptr;       // T * n_embd
    int32_t* h_out_ = nullptr;   int32_t* m_out_ = nullptr;     // T argmax ids
    uint32_t* h_nf_ = nullptr;   uint32_t* m_nf_ = nullptr;     // #879: kVerifyMaxT * kNonfiniteBlocks flags
    int nf_row_ = -1;                     ///< nonfinite_row()
    int64_t nf_pos_ = -1;                 ///< nonfinite_pos()
    uint32_t nf_mask_ = 0;                ///< nonfinite_in(): bit t = row t
    void note_nonfinite(int T, const int64_t* pos_b);   ///< the window's flags (+ STRATA_NAN_INJECT) -> nf_row_
    int32_t* h_route_ = nullptr; int32_t* m_route_ = nullptr;   // n_layers * T * k: each layer's ids (add_routing)
    bool route_rec_ = false;
    // STRATA_ROUTE_DUMP / STRATA_ROUTE_REPLAY (verify.cpp's RouteTrace): the routed weights beside h_route_'s ids, the
    // replayed ids and weights (n_layers * T * k each), and the request and position the last window was in
    float* h_rw_ = nullptr;      float* m_rw_ = nullptr;
    int32_t* h_rpl_ids_ = nullptr; int32_t* m_rpl_ids_ = nullptr;
    float* h_rpl_w_ = nullptr;   float* m_rpl_w_ = nullptr;
    bool rt_dump_ = false, rt_replay_ = false;
    int rt_req_ = -1;
    int64_t rt_pos0_ = INT64_MAX;
    strata::gpu::Event* commit_done_ = nullptr;   // recorded after an async commit (set_commit_async)
    bool commit_pending_ = false;
    // the routed experts' protocol with the host: the mapped doorbell, ids, weights, CPU rows, flags and plans, the
    // copy engine's stream, the experts' device rows (moe_window.hpp, the plain engine's too).  A device whose running
    // kernels do not see the host's writes runs a window as segments the host launches (win_.segmented()).
    MoeWindow win_;
    bool split_ = false;   // opt-in (--spec-split): exact but slower, see the overlap study
    int groups_[9] = {};

    // device
    void* arena_ = nullptr;
    int32_t *tok_ = nullptr, *step_ = nullptr, *pos_ = nullptr, *commit_ = nullptr;
    float *ple_ = nullptr, *emb_ = nullptr, *R_ = nullptr, *mixed_ = nullptr, *bo_ = nullptr;
    float *inj_ = nullptr, *inj2_ = nullptr, *lo_ = nullptr, *rs_ = nullptr, *xn_ = nullptr;
    uint8_t* xq_ = nullptr;                                   // T columns of q8_1
    float *qkv_L_ = nullptr, *h_L_ = nullptr, *gate_L_ = nullptr, *beta_L_ = nullptr;   // per GDN layer
    float *z_ = nullptr, *y_ = nullptr, *y_dummy_ = nullptr;
    float *qfull_ = nullptr, *qcur_ = nullptr, *kcur_ = nullptr, *vcur_ = nullptr, *idx_raw_L_ = nullptr;
    float *qidx_ = nullptr, *scores_ = nullptr, *attn_ = nullptr, *attn32_ = nullptr, *attn_scratch_ = nullptr;
    float* tail_snap_ = nullptr;                              // per QSA layer
    int32_t* sel_ = nullptr;
    float *logits_ = nullptr, *w_ = nullptr, *shared_ = nullptr;
    int32_t* ids_ = nullptr;
    float* nat_xr_ = nullptr;     // APR: the Hadamard-rotated expert input (the shared expert and the CPU keep xm); nullptr
                                  // without a folded gate/up
    const float* had_sx_ = nullptr;   // APR: this device's sign vectors of the expert input (n_embd) and of h (n_ff)
    const float* had_sh_ = nullptr;
    float *head_mixed_ = nullptr, *head_inj_ = nullptr, *head_logits_ = nullptr;
    int32_t* one_ = nullptr;
    float* ple_key_ = nullptr;   ///< the window rows' PLE key and value projections (ple_batch)
    float* ple_val_ = nullptr;   ///< device 1: a one-token window's n_keep (it commits itself)
    float *sh_gate_ = nullptr, *sh_up_ = nullptr, *sh_g_ = nullptr;
    float* hist_snap_ = nullptr;                              // T * NG_HIST * NG_HC_DIM
    int64_t cap_ = 0, max_blocks_ = 0, attn_scratch_floats_ = 0;
};

}  // namespace strata::core
