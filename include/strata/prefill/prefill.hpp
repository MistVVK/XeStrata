// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/prefill/prefill.hpp - plan v0.3 P5: batched prompt processing.
//
// The prompt's positions [pos0, pos0 + n) are processed in chunks of `chunk` tokens through all 48 layers, leaving
// the session state (GDN recurrence and conv state, QSA KV pools and indexer, PLE history) where the token path
// would have left it; the decode loop then continues with the next token.  Per layer: the projections are
// tensor-core GEMMs (quantized weights dequantized to FP16 on the fly, BF16 weights as they are), the recurrences
// walk the chunk inside one kernel, and the routed experts are grouped by expert: resident ones are read from the
// VRAM tier, the others streamed from the host arena through a pinned ring on a copy stream.
//
// Requires the native weights (`--native`): every quantized projection must carry its GGUF blocks.
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"
#include "strata/platform/env.hpp"

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace strata::kernels::cpu { class ExpertPool; }
namespace strata::gpu { struct Event; }

namespace strata::prefill {

struct PrefillStats {
    int64_t tokens = 0;
    int64_t chunks = 0;
    double ms_total = 0;
    double ms_experts_host = 0;     ///< host time staging non-resident experts
    int64_t experts_streamed = 0;   ///< expert blobs copied host -> device
    int64_t experts_dma = 0;        ///< ...of which straight from the pinned arena (no CPU copy)
    int64_t experts_resident = 0;   ///< expert-layer groups served from the VRAM tier
    int64_t experts_cpu = 0;        ///< ...computed on the CPU pool instead of streamed (STRATA_PREFILL_CPU_SHARE)
    int64_t experts_helper = 0;     ///< ...computed on a helper GPU from its own cache instead of streamed (set_helper)
    double cpu_share = 0;           ///< ...the share of the streamed ones it took last (measured with `auto`)
    double ms_ple = 0;
    /// The prompts' routings, (token, k) pairs per model layer x expert (n_layers x n_expert; empty before the first
    /// MoE layer), and the k of a token: what the decode after a prompt is likely to route to, set against the cache
    std::vector<int64_t> routed;
    int64_t routed_k = 0;
};

/// --prefill-cpu-share: the CPU pool's share of a short chunk's streamed experts, fixed (0..1; negative: measured,
/// the default) where the default arms the share at all.  STRATA_PREFILL_CPU_SHARE, when set, wins.  Before any prompt.
void set_cpu_share(double share);
/// The share in use: STRATA_PREFILL_CPU_SHARE's, else set_cpu_share's; -1 measured.
double cpu_share_setting();

/// The prompt chunks --prefill auto picks from, the largest first, for both engines (generate.cpp's plan_lend and
/// PlainModel's): every chunk streams again each expert it routes to, so fewer chunks stream fewer experts.  32768 and
/// 16384 after upstream ae652b2: on the B70 a 26,292-token IQ3_S prompt read at 1,080 tok/s in 8192-token chunks
/// (50,356 experts streamed), 1,303 in 16384 and 1,545 in one chunk (17,728), the same logits.
inline constexpr int64_t kAutoChunks[] = {32768, 16384, 8192, 6144, 4096, 3072, 2048, 1024, 512, 256};
/// The largest of kAutoChunks that `take(chunk)` accepts (0: none), at most STRATA_PREFILL_AUTO_MAX (32768 unset) and,
/// past 8192, none above `max_context`.
template <class Take>
int64_t auto_chunk(int64_t max_context, Take&& take) {
    static const int64_t auto_max = [] {
        const char* v = strata::env_get("PREFILL_AUTO_MAX");
        return v ? (int64_t) std::strtoll(v, nullptr, 10) : (int64_t) 32768;
    }();
    for (const int64_t c : kAutoChunks) {
        if (c > auto_max || (c > 8192 && c > max_context)) continue;
        if (take(c)) return c;
    }
    return 0;
}
/// The percent of the expert tier's slots a prompt chunk's buffers may borrow (a lent slot's expert is streamed during
/// the prompt and refilled after it): STRATA_PREFILL_LEND_PCT, else `fallback` (the engine's own, measured).
inline int64_t lend_pct(int64_t fallback) {
    const char* v = strata::env_get("PREFILL_LEND_PCT");
    return v ? (int64_t) std::strtol(v, nullptr, 10) : fallback;
}

}  // namespace strata::prefill
namespace strata::core { class MtpDrafter; class RemoteExperts; }
namespace strata::prefill {

class Prefill;

/// LAYER SPLIT: what a prompt path's stage asks of the stage after it - a chunk's rows at the split point with the
/// chunk's tokens and position, and the end of that chunk - and nothing else, so a next stage that shares nothing
/// else with this one (another process, another machine) can stand behind it.
class PrefillLink {
public:
    virtual ~PrefillLink() = default;
    /// The next stage reads the chunk: `rows` (T x hc*n_embd floats, pinned host memory, untouched until the
    /// `wait` after the next `chunk`), `tokens[0..T)` at positions p0...  Returns once it has started.
    virtual bool chunk(const float* rows, const int64_t* tokens, int64_t T, int64_t p0, std::string& err) = 0;
    /// Waits for the last chunk the next stage was given (none: true); false with `err` when it failed.
    virtual bool wait(std::string& err) = 0;
    /// STRATA_HANDOFF_PIPE: `chunk`, with `rows` still landing in `pieces` row pieces (handoff_piece), piece i done
    /// when `landed[i]` is (events of this stage's device).  A link that cannot start on part of the rows waits for
    /// them all here.
    virtual bool chunk_pieces(const float* rows, gpu::Event* const* landed, int pieces, const int64_t* tokens,
                              int64_t T, int64_t p0, std::string& err);
    /// The next stage when it is in this process (null: elsewhere).  The layer-major prefill drives it directly.
    virtual Prefill* local_next() const { return nullptr; }
};

/// STRATA_HANDOFF_PIPE: rows [r0, r0 + nr) of piece i of a T-row chunk in P pieces (both sides of a hand-off).
void handoff_piece(int64_t T, int P, int i, int64_t& r0, int64_t& nr);

/// The next stage in this process: its `run` on a thread, so it reads chunk c while this stage reads chunk c + 1.
class InProcessPrefill final : public PrefillLink {
public:
    InProcessPrefill();
    ~InProcessPrefill() override;
    InProcessPrefill(const InProcessPrefill&) = delete;
    InProcessPrefill& operator=(const InProcessPrefill&) = delete;
    void bind(Prefill* next) { next_ = next; }
    bool chunk(const float* rows, const int64_t* tokens, int64_t T, int64_t p0, std::string& err) override;
    bool wait(std::string& err) override;
    bool chunk_pieces(const float* rows, gpu::Event* const* landed, int pieces, const int64_t* tokens, int64_t T,
                      int64_t p0, std::string& err) override;
    Prefill* local_next() const override { return next_; }

private:
    bool start(const int64_t* tokens, int64_t T, int64_t p0);
    Prefill* next_ = nullptr;
    std::future<bool> run_;
    std::string err_;
    struct Worker;
    std::unique_ptr<Worker> worker_;   ///< STRATA_HANDOFF_PIPE: the next stage's chunks' thread (started once)
};

class Prefill {
public:
    /// The CPU expert pool (decode's, idle while a prompt is read).  With STRATA_PREFILL_CPU_SHARE set, a chunk below
    /// the streamed-walk size hands it the non-resident experts routed by at most MAXT of its tokens, fewest first, up
    /// to a share of the experts it would stream (`auto`: measured, where both sides end together); their rows go to
    /// the tail of the experts' outputs (upstream 978d3558).  Not bit-identical to the GPU's rows (the CPU's own
    /// activation format).  Unset (default) or null: every expert on the GPU.  Only for a pool no other thread runs
    /// meanwhile (no batch slots).  Set before `init`.
    void set_cpu_pool(kernels::cpu::ExpertPool* pool) { cpu_pool_ = pool; }

    /// A helper GPU's expert cache (RemoteExperts: --expert-cache-device1, or --layer-split auto's helper).  A chunk
    /// below the streamed-walk size hands it the experts this GPU does not hold and it does, routed by at most
    /// `max_rows` of the chunk's tokens: it computes their rows from its VRAM (int8 products) while this GPU streams
    /// the others, and they come back over the two links through host memory registered on both GPUs; their rows go
    /// after the CPU pool's.  Not bit-identical to this GPU's rows (another GPU's products).  Only for a helper that
    /// computes nothing else meanwhile (no batch slots).  False with `err` when the helper cannot take it (no int8
    /// products there, no memory); the prompt path then streams every expert as before.
    bool set_helper(core::RemoteExperts* helper, int64_t max_rows, std::string& err);

private:
    kernels::cpu::ExpertPool* cpu_pool_ = nullptr;
    struct HelperShare;
    std::unique_ptr<HelperShare> helper_;

public:
    /// E-9 (upstream c6c6594): the draft layer's K/V for prompt cells [cell0, cell0 + n) from their final residual
    /// rows `R_rows` (device) and `next_tokens` (host: the token at cell+1), in batches through this path's GEMMs
    /// and its idle scratch - call it from on_chunk.  false with `err` empty: not applicable here (a streamed K/V,
    /// another device, too little scratch; STRATA_MTP_BATCH=0), the caller runs the drafter's own pass.  Not
    /// bit-identical to that pass (FP16 products instead of Q8_1 activations): the drafts may differ, never the
    /// target's logits.  `why` (optional): on a decline (false, `err` empty) the reason, a static string.
    bool draft_kv(core::MtpDrafter& mtp, const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0,
                  std::string& err, const char** why = nullptr);
    Prefill();
    ~Prefill();
    Prefill(const Prefill&) = delete;
    Prefill& operator=(const Prefill&) = delete;

    /// Frees every buffer, stream and event `init` made (as the destructor does) and starts over empty, so `init` can
    /// run again - with a smaller chunk when the first one did not fit.  The stage range (`set_stage`) and the
    /// callbacks stay.  The device `init` ran on must be current.
    void reset();

    /// `host_res`: the static residency table (n_layers x n_expert, slot or -1) or null; `cache` its slots.
    /// `borrow`/`borrow_bytes`: device memory to carve every buffer from (the top slots of the expert cache,
    /// lent for the prompt and refilled after it); null = allocate normally.
    bool init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
              core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
              void* stream, std::string& err, void* borrow = nullptr, uint64_t borrow_bytes = 0);

    /// With borrowed buffers: lay them out again for chunks of `chunk` tokens (at most `init`'s) in `borrow` - a
    /// request lends only the slots its prompt needs.  The stream must be idle (between prompts).
    bool relayout(int64_t chunk, void* borrow, uint64_t borrow_bytes, std::string& err);
    int64_t chunk() const;

    /// The share of the streamed experts' bytes DMA-able straight from pinned RAM (1 = all).  Sizes the streamed
    /// ring (a big one only pays when the copy engine, not the host copies, is the limit); set before bytes_needed.
    static void set_pinned_share(double share);
    static double pinned_share();
    /// The int8 products' group scratch sized for one group's rows (on) or the chunk's (off, the default): the chunk
    /// planner turns it on where the chunk-sized buffers do not fit.  Before any prompt; bytes_needed counts it.
    static void set_group_cap(bool on);
    static bool group_cap();
    /// #340: the streamed ring's slot count for chunks that stream every expert, instead of the pinned-share rule
    /// (0 = that rule). Set before any `bytes_needed`/`init` (both count the ring); STRATA_PREFILL_RING still wins.
    static void set_ring_override(int slots);

    /// Experts a grouped product takes (1-64, default 16): each holds 9.4 MiB of dequantized FP16 weights in the
    /// region; more of them make fewer, fuller launches.  Set before bytes_needed.
    static void set_expert_group(int g);
    static int expert_group();

    /// Device bytes `init` needs for a chunk of `chunk` tokens (what a borrowed region must hold).
    static uint64_t bytes_needed(const core::ModelGeometry& g, const core::SessionState& ss, int64_t chunk);

    /// Positions [pos0, pos0 + n) holding `tokens`; `ss.ple_prev` must be the two tokens before pos0 (oldest
    /// first, -1 for none) and is advanced to the last two of these.
    bool run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);

    const PrefillStats& stats() const { return stats_; }

    /// Plan v0.3 P6: called after every chunk with the chunk's final multi-stream residual rows (device,
    /// T x hc*n_embd, valid until the next chunk) and the chunk's first position; the MTP draft layer builds its
    /// K/V from them.  The prefill stream is synchronized before the call.
    std::function<bool(const float* R_rows, int64_t T, int64_t pos0, std::string& err)> on_chunk;

    /// The DFlash2 drafter's prompt taps: the completed output (the mean of the hyper-connection streams) of these
    /// layers, for the rows at or after `first_pos`, gathered into a staging of `cap_rows` rows and reported through
    /// on_taps after each chunk (the prefill stream synchronized).  Set before init; empty allocates nothing.
    void set_taps(const std::vector<int>& layers, int64_t first_pos, int64_t cap_rows) {
        tap_layers_ = layers; tap_first_ = first_pos; tap_cap_ = cap_rows;
    }
    std::function<bool(const float* taps, int64_t rows, int64_t pos0, std::string& err)> on_taps;

    /// Layer split: called by every stage when it has read a chunk, with the position reached, while its own state
    /// is still at that chunk's end (its stream synchronized; the last stage calls it just before `on_chunk`).  An
    /// earlier stage is a chunk or more ahead of the last one by the time `on_chunk` runs, so this is where a
    /// mid-prompt checkpoint takes each stage's part.  Runs on that stage's thread, with its device current.
    std::function<bool(int64_t done, std::string& err)> on_stage_chunk;

    /// Checked before every chunk: true stops the prompt early (`run` returns false with err "cancelled").
    std::function<bool()> should_stop;

    /// The vision path: HOST rows (n_embd floats) indexed by absolute position, read in place of the token
    /// embedding where non-null (an image's <|image_pad|> cells).  Null (default): every position embeds its token.
    const float* const* embd_rows = nullptr;

    /// LAYER SPLIT (multi-GPU): this prompt path runs layers [layer_begin, layer_end) (-1: to the last) on the
    /// device `init` runs on.  A stage that does not start at layer 0 reads each chunk's residual rows from the
    /// previous stage instead of embedding the tokens; a stage that does not end at the last layer copies its rows
    /// to pinned host buffers (two, allocated by `init`) and runs `next` on them - on a thread, so the next stage
    /// reads chunk c while this one reads chunk c + 1.  `on_chunk` belongs on the last stage.  Set before `init`.
    /// Everything goes to `next` through a PrefillLink (an InProcessPrefill here).
    void set_stage(int64_t layer_begin, int64_t layer_end, Prefill* next) {
        stage_lb_ = layer_begin; stage_le_ = layer_end;
        local_next_.bind(next);
        next_ = next != nullptr ? &local_next_ : nullptr;
    }
    /// The previous stage in this process hands the rows of the chunk this stage reads next (pinned host memory,
    /// T x hc*n_embd floats, untouched until that chunk is read).  InProcessPrefill::chunk.
    /// STRATA_HANDOFF_PIPE: `landed` (`pieces` events of the previous stage's device) say when each row piece is in.
    void accept_rows(const float* rows, gpu::Event* const* landed = nullptr, int pieces = 0) {
        hand_in_ = rows;
        hand_ev_ = landed;
        hand_pieces_ = pieces;
    }

    /// Layer-major prefill (STRATA_PREFILL_LAYER_MAJOR=1): true while `on_chunk` reports a sub-chunk before the end of
    /// its super-chunk.  The session state is already at the super-chunk's end then, so no checkpoint may be taken.
    /// On a layer split it is the last stage's (the one that reports the rows), asked through the first.
    bool layer_major_mid() const {
        const Prefill* p = this;
        while (p->lm_next() != nullptr) p = p->lm_next();
        return p->lm_mid_;
    }

private:
    /// The default walk: each chunk through all the stage's layers.
    bool run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);
    // Layer-major prefill: the super-chunk's buffers for an n-token prompt (its size, 0 = not this prompt: `why`), the
    // run, and the buffers given back after it.  `cap` (> 0): at most that many tokens.
    int64_t lm_prepare(int64_t n, std::string& why, int64_t cap = 0);
    int64_t lm_prepare_chain(int64_t n, std::string& why);
    void lm_release_chain();
    bool run_layer_major(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err);
    void lm_release();
    bool lm_drain(std::string& err);
    Prefill* lm_next() const { return next_ != nullptr ? next_->local_next() : nullptr; }
    bool lm_mid_ = false;
    std::future<bool> lm_next_run_;   ///< a layer split: the next stage's run of the last super-chunk handed on
    std::string lm_next_err_;
    int64_t stage_lb_ = 0, stage_le_ = -1;
    std::vector<int> tap_layers_, tap_local_, tap_slot_;   ///< set_taps: the drafter's tap layers (this stage's share)
    int64_t tap_first_ = 0, tap_cap_ = 0;
    float* tap_dev_ = nullptr;       ///< [tap_cap_][ntap][n_embd] staging (null: no taps)
    PrefillLink* next_ = nullptr;       ///< set_stage: the next stage, through this link only
    InProcessPrefill local_next_;
    const float* hand_in_ = nullptr;    ///< the previous stage's rows of the chunk being read (host, pinned)
    gpu::Event* const* hand_ev_ = nullptr;   ///< STRATA_HANDOFF_PIPE: when each piece of them has landed (or null)
    int hand_pieces_ = 0;
    bool carve(std::size_t T, void* alloc);   // the device buffers of a chunk (prefill.cpp's Alloc)
    void release();                          // the destructor's cleanup (also `reset`'s)
    struct Impl;
    std::unique_ptr<Impl> impl_;
    PrefillStats stats_;
};

/// What multiplies the prompt path's experts on the current GPU, for the start's report: "int8 (WMMA)" and the like
/// (moe_mmq.hpp), with the layers of types the int8 products lack said, or "FP16".  Call once the expert layout is read.
std::string experts_path();

}  // namespace strata::prefill
