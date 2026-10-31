// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/core/plain_model.hpp - an MoE model with one residual stream, run from its GGUF.
//
// qwen35moe (Ornith-1.5: Gated DeltaNet layers and full-attention layers, a shared expert) and qwen3moe (llm-jp-4.1:
// full attention in every block, no shared expert) - the models whose block is
//
//     x += mixer(rms_norm(x))            the mixer a GDN layer or a gated, partly rotated full attention
//     x += moe(rms_norm(x))              routed experts (softmax top-k, renormalized) + the gated shared expert
//
// and whose head is rms_norm, then the output projection.  Qwen3.8 (four residual streams, the QSA indexer, PLE) keeps
// its own engine.
//
// qwen35 (Qwen3.5-9B) is qwen35moe's block with a dense SwiGLU in place of the MoE (`dense_`): every weight in VRAM, a
// window one graph with nothing for the host to answer, the prompt's buffers of their own; no tiers, no CPU pool.
//
// glm5-next (GLM-5.3-Flash, plain_glm5.cpp) runs here too, with its own block: four mHC streams read before and written
// after every mixer and FFN, KDA (per-channel decay) and DSA (a k-pool indexer, the MLA attention over its selection)
// mixers, leading dense FFN layers, a sigmoid router and the clamped SwiGLU; the expert tiers and the windows as above.
//
// The dense weights are uploaded once in their GGUF form and read by the native kernels.  The routed experts go
// through the engine's expert tiers, as Qwen3.8's do: a VRAM cache of slots (expert_cache) refilled between windows by
// the adaptive rule, the GGUF read in place (FileExpertSource: the page cache, the drive), the CPU pool for what the
// VRAM does not hold, all in one window protocol (MoeWindow).  A window is 1..max_t tokens at consecutive positions,
// recorded once per size as a graph.
//
// A qwen35moe file with its multi-token-prediction layer (`blk.<n_layer>.nextn.*`, Ornith's official K-quants) drafts
// with it (`spec`): that layer is a full-attention block of the trunk's form with its own K/V, after a projection of
// [rms_norm(embedding of token p) | rms_norm(the trunk's final normed row at p - 1)] (llama.cpp's graph_mtp), and its
// routed experts are one more MoE layer of the tiers.  A window checks the last token and the drafts; the recurrent
// states are copied after each row but the last, and put back to the last accepted row's copy when a draft is wrong.
#pragma once

#include "strata/artifact/model_spec.hpp"
#include "strata/core/draft_head.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/moe_window.hpp"
#include "strata/core/vmm.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/spec/draft_policy.hpp"

#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace strata {
class GgufModel;
}

namespace strata::core {

struct PlainOptions {
    std::string gguf;               ///< any shard of the model
    int64_t max_cells = 4096;       ///< the context the K/V cache holds
    /// --kv int8: the attention layers' K and V as int8 codes with an FP16 scale per 64 values (kv_q8.hpp), half the
    /// FP16 cache's bytes; the answers differ slightly.  Not GLM-5.3-Flash's DSA caches.
    bool kv_int8 = false;
    /// --kv q4_0 (1): K and V as Q4_0 after the 256-point Hadamard rotation (kv_q4.hpp, the query rotated the same way
    /// and the output back); --kv k8v4 (2): K in INT8 (kv_int8's codes and scales), V rotated Q4_0.  head_dim 256 only;
    /// not GLM-5.3-Flash's caches.
    int kv_q4 = 0;
    /// --kv-ram on|off (GLM-5.3-Flash): the DSA layers' MLA latent cells in host memory instead of VRAM, which goes to
    /// the expert tier: a window copies the cells each token selected into VRAM once (glm5_latent_gather), a prompt
    /// chunk the layer's cells up to it (Project Maya #86's idea, written anew: there the cells live in VRAM slots).
    /// On by default: with 128K cells the B70's tier went 2,150 -> 2,322 slots and decode 3.84 -> 3.94 tok/s at 56 GB
    /// of RAM; the RTX 4070 started at all (it had 161 MiB left for the tier), and at 64K went 86 -> 172 slots, its
    /// prompt 3.3 -> 7.1 tok/s and decode 2.32 -> 2.30 (UD-IQ3_XXS, 1K + 256, 2026-10-26).  The other models: off.
    bool kv_ram = true;
    /// --vision (qwen35, qwen35moe): pictures among the prompt's tokens (plain_generate's GENI): their rows in place
    /// of the token embeddings (set_embeddings) and every cell's M-RoPE position from a table (set_positions)
    bool vision = false;
    /// An image model's text encoder (encode: MOSS-Image reads Qwen3.5-9B's states): no head, no decode attention
    /// timed; the prompt runs in chunks only.  A dense model (qwen35) on one GPU.
    bool encoder = false;
    /// An encoder's block matrices left in the mapped file (block_spans): a BlockSource puts each block's on the device
    /// as the prompt walks the blocks (strata-image's LayerStream, within its budget)
    bool stream_blocks = false;
    /// --vram-elastic (#533; --serve, one GPU with virtual memory): the expert slots in physical segments of
    /// vram_segment_mib that `vram` gives back to the driver and maps again between requests
    bool vram_elastic = false;
    int64_t vram_segment_mib = 512;
    int max_t = 8;                  ///< the largest window
    /// --batch N (--serve): conversations decoded together, one row each in a window (plain_serve's batch slots), each
    /// slot its own copy of the per-sequence state; at most max_t, as many as take half the VRAM free before the
    /// expert tier at most.  Not GLM-5.3-Flash's nor with --vision (one table of rotary positions) for now.
    int batch_slots = 0;
    /// --kv-grow (1) / --no-kv-grow (0) (--serve): the live session's K/V in virtual memory, physical memory only for
    /// the cells the requests reach, the expert tier (in --vram-elastic's segments) giving VRAM up for them and taking
    /// it back after a short request; -1: on where the device's virtual memory pages are 2 MiB or more, as
    /// Qwen3.8-Flash-Next's.  One GPU, a model with experts, no --batch.
    int kv_grow = -1;
    int pool_workers = 0;           ///< CPU pool workers (0: every physical core but the first)
    int pool_tasks = 0;             ///< --pool-tasks: the pool's tasks a phase (0: 3 a participating thread)
    bool pool_host_works = true;    ///< --no-host-worker: false, the host thread does not join the pool's drain
    int64_t reserve_mib = -1;       ///< VRAM left free after the expert slots (-1: from the device's memory)
    int64_t reserve_later_mib = -1; ///< ... on a layer split's later GPUs (-1: reserve_mib)
    int64_t vram_free_mib = -1;     ///< the free VRAM the tier is sized from at most (-1: what the device reports)
    int64_t expert_slots = -1;      ///< the VRAM tier's slots at most, over all the layers (-1: what fits)
    int adapt_every = 4;            ///< windows between adaptive rounds (0: off)
    bool adapt_every_given = false; ///< --adapt-every was given: its period stays (no auto_adapt_period)
    int adapt_swaps = 96;           ///< the most experts one round moves into the VRAM tier
    /// ... and at most this many MiB of experts for each token the windows read since the last round: the copies share
    /// the RAM's bandwidth with the missed experts.  48: on the RX 9060 XT with a Ryzen 7 5700X (DDR4, 2026-10-12),
    /// llm-jp-4.1 (4.6 MB experts, a token a window) went 31.3 -> 34.3 tok/s at 8 GB (96 a round had been 100 MB a token)
    /// and 58.6 -> 60.0 at 16 GB (32: 35.3 and 58.1); Ornith Q6_K (2.5 MB, MTP windows) stays at its 96 a round, 21-24 MB
    /// a token.  On the B70 with an i7-14700 (DDR5) the bound changed nothing (llm-jp 8 GB 41.5 / 41.6).  0: none.
    int adapt_mib_per_token = 48;
    float adapt_decay = 0.7f;       ///< the routing counts are multiplied by this after each round
    float adapt_min_gain = 1.5f;    ///< an expert moves in only when its count is this much over the one it evicts
    /// The experts read from the GGUF in place (the page cache, the drive) instead of loaded into pinned RAM: the
    /// low-RAM mode, taken by itself when the RAM available is less than the experts.  It has no PCIe share.
    bool mmap_experts = false;
    /// generate.cpp's resident low-RAM mode: the experts read in place, and those no VRAM tier holds copied into RAM
    /// in the profile's order (finish_experts) while they fit `resident_gib` GiB (< 0: --resident-experts, as many as
    /// the RAM allows; 0: none beyond what the RAM's own shortage picks); `resident_pageable` (--resident-cpu-experts):
    /// the copy in ordinary RAM, not page-locked
    double resident_gib = 0;
    bool resident_pageable = false;
    /// --shared-expert-arena FILE (Linux): the experts' RAM shared with the other engines that name it (the arena's
    /// first loader fills it, the rest map it), as Qwen3.8-Flash-Next's
    std::string shared_arena;
    bool pcie_split = true;         ///< split the missed experts between the CPU and PCIe by measured times
    /// --pcie-frac F: that share of the missed experts over PCIe in every window, instead of the measured split and the
    /// decode's first windows' choice (< 0: measured)
    double pcie_frac = -1.0;
    /// A ranking of (MoE layer, expert) pairs (expert_cache.hpp's profile format) the VRAM tier is filled from at load,
    /// each layer up to its slots; empty: the tier starts empty and the adaptive rule fills it.
    std::string profile;
    /// The MTP layer's drafts over a token subset (int32 ids, tools/draft_vocab.py's data/draft_vocab*.bin): a head of
    /// those rows only, so a draft reads a part of the vocabulary; a token outside it is never drafted.  Empty: the
    /// whole head.
    std::string draft_vocab;
    /// The rope scaling, as llama.cpp's options: "yarn" / "linear" / "none" (empty: the file's), the factor (0: the
    /// file's), the trained context (0: the file's, else its context length).  llm-jp-4.1 at 256K: yarn, 4, 65536.
    std::string rope_scaling;
    double rope_scale = 0;
    double yarn_orig_ctx = 0;
    /// YaRN's other knobs, as llama.cpp's (generate.cpp's --yarn-*): the correction depth (< 0: 1 for yarn, else 0),
    /// the magnitude scale and the two ramps' bounds
    double yarn_ext_factor = -1.0, yarn_attn_factor = 1.0, yarn_beta_fast = 32.0, yarn_beta_slow = 1.0;
    /// The routed experts written here, a record a token and MoE layer as generate.cpp's --dump-routing writes them
    /// (int32 layer, int32 k, k int32 ids, k float weights, 0 here): tools/make_profile.py reads them.  Empty: none.
    std::string dump_routing;
    double rope_freq_base = 0;      ///< the rotation's base (0: the file's)
    double rope_freq_scale = 0;     ///< ... and its frequency scale, as llama.cpp's --rope-freq-scale (0: none)
    /// The most tokens a prompt chunk takes through the batched products (`prompt`; a shorter prompt, one chunk of its
    /// length); 0: the prompt runs in windows.
    int64_t prompt_chunk = 4096;
    bool prompt_chunk_given = false;   ///< --prompt-chunk was given: no chunk is chosen (auto_prompt_chunk)
    /// Speculative decoding with the model's MTP layer: windows of up to this many tokens, the last accepted one and
    /// spec - 1 drafts (0 or 1: off; a model without the layer: off).  Greedy requests only.  3 by default: two
    /// drafts were the fastest of one to three on the RX 9060 XT (16 and 8 GB), the B70 and the RTX 4070 (Ornith Q6_K
    /// and Q4_K_M, 2026-10-12).
    int spec = 3;
    /// --spec-window: how many of the drafts a window checks.  "all": every draft made (the fixed window of spec
    /// tokens); "auto": the drafts kept while the tokens each is expected to add, at its calibrated acceptance, raise
    /// the window's expected tokens per ms at the measured window costs (strata::spec::DraftPolicy, as Qwen3.8-Flash-
    /// Next's engine); "min-p": the drafts while each is at least spec_min_p likely under the MTP layer.  "all" by
    /// default: auto decoded as fast at spec 3, no faster (Ornith-1.5 Q6_K, 256 tokens: B70 95.5 / 97.7 against 96.5 /
    /// 95.8 tok/s, RTX 4070 80.0 / 84.2 against 79.4 / 84.7; GLM-5.3-Flash at 56 GB of RAM on the B70 4.17 against
    /// 4.13, 2026-10-26).
    std::string spec_window = "all";
    double spec_min_p = 0.0;
    /// A layer split across the engine's GPUs (--layer-split): GPU 0 runs the blocks before split_at[0], GPU 1 those
    /// from there to split_at[1], and so on, the last GPU the rest, the head and the MTP layer; `split_auto`: the points
    /// chosen at load from each GPU's free memory.  Empty and false: one GPU, the current one.
    std::vector<int64_t> split_at;
    bool split_auto = false;
    /// The engine's GPUs the later stages take (--split-device), in order; empty: the other cards
    std::vector<int> split_devices;
};

// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding): one object, its members grouped by what they serve
class PlainModel {
public:
    PlainModel();
    ~PlainModel();
    PlainModel(const PlainModel&) = delete;
    PlainModel& operator=(const PlainModel&) = delete;

    /// Reads the model's shape, uploads its dense weights, opens its experts and sizes the VRAM tier from what the
    /// device has left.  False, with the reason, for a model this engine does not run.
    bool load(const PlainOptions& opt, std::string& err);
    const ModelSpec& spec() const { return spec_; }
    int64_t vocab() const { return n_vocab_; }
    /// The model carries a recurrent state (GDN layers): a sequence can be extended, not cut back to a prefix.
    bool recurrent() const { return !gdn_state_.empty(); }
    int64_t max_cells() const { return opt_.max_cells; }
    /// The prompt's chunk (0: the prompt runs in windows; GLM-5.3-Flash's for now, whatever the options asked).
    int64_t prompt_chunk() const { return opt_.prompt_chunk; }
    /// --vision: the M-RoPE positions (t, h, w) of cells 0 .. n - 1, `pos3` 3 n of them on the host
    /// (kernels/mrope.hpp's table, every stage's; the cells after n keep theirs).  Between requests.
    bool set_positions(const int32_t* pos3, int64_t n, std::string& err);
    /// --vision: (*rows)[p], n_embd floats, in place of the embedding of the token at position p where it is not null
    /// (a picture's rows; null `rows`: none).  The vector outlives the prompt it is set for.
    void set_embeddings(const std::vector<const float*>* rows);

    /// Zeroes the recurrent states and the K/V cache: the next window starts a sequence at position 0.
    bool reset(std::string& err);
    /// The state of the first `tokens` tokens on the host (a conversation set aside: plain_serve's conversation
    /// cache and session files): every attention layer's K/V pages that hold them, every recurrent layer's state and
    /// convolution history, GLM's DSA caches.  The device is synchronized first; restore puts it back as it was, the
    /// pending draft dropped.
    struct Snapshot {
        int64_t tokens = 0;
        std::vector<std::vector<uint8_t>> parts;
        size_t bytes() const;
    };
    bool save_state(int64_t tokens, Snapshot& snap, std::string& err);
    bool restore_state(const Snapshot& snap, std::string& err);
    /// The recurrent layers' part of it alone (plain_serve's mid-prompt checkpoints): every stage's recurrent states and
    /// convolution histories as `tokens` tokens left them.  Restoring one is right only while the K/V cells below
    /// `tokens` still hold the same tokens; nothing for a model without a recurrent state.
    bool save_recurrent(int64_t tokens, Snapshot& snap, std::string& err);
    bool restore_recurrent(const Snapshot& snap, std::string& err);
    /// --batch: the slots the load gave (on every stage), and the state of the first `tokens` tokens copied on the
    /// device from session `from` to session `to` (-1: the live one, else a slot), the recurrent parts alone with
    /// `recurrent_only`; save_slot is save_state's for slot b.  Into the live session, the pending draft is dropped.
    int batch_slot_count() const;
    bool copy_session(int from, int to, int64_t tokens, bool recurrent_only, std::string& err);
    bool save_slot(int b, int64_t tokens, Snapshot& snap, std::string& err);
    /// A batch window: row t is token tokens[t] of slot slot[t] (ascending, each once) at its position pos[t],
    /// out[t] the token after it, drawn with sp[t] (the row's position its counter).  T <= max_t.
    bool forward_slots(const int32_t* tokens, const int* slot, const int64_t* pos,
                       const strata::kernels::SamplerParams* sp, int T, int32_t* out, std::string& err);
    /// The bytes save_state would take for `tokens` tokens
    size_t state_bytes(int64_t tokens) const;
    /// Tokens [0, T) at positions pos .. pos + T - 1 (T <= max_t): out[t] is the greedy token after token t, and with
    /// `logits` the T rows of n_vocab logits.  Positions must follow the previous window's.
    /// `defer`: the window's rows are not yet known to be kept (a verify window): spec_step counts the ones it keeps
    /// (after_rows), else every row counts.
    bool forward(const int32_t* tokens, int T, int64_t pos, int32_t* out, float* logits, std::string& err,
                 bool defer = false);
    /// The prompt: tokens [0, n) at positions pos .. pos + n - 1 in chunks of `prompt_chunk` tokens, every projection a
    /// matrix product and the routed experts grouped by expert, all on the GPU (src/prefill/plain_prompt.cpp, in the
    /// prompt path's library).  `out`: the greedy token after the last; `logits`: its row (or null).  The state is left
    /// where the windows would have left it.  False with `err` empty when the VRAM tier cannot lend the prompt's
    /// buffers even for a short chunk (a 4 GB card): nothing ran, and the caller reads the prompt through the windows.
    bool prompt(const int32_t* tokens, int64_t n, int64_t pos, int32_t* out, float* logits, std::string& err);
    /// Told after each chunk prompt() reads: the tokens read so far and the prompt's (its `n`); empty: nothing.  The
    /// serve loop prints it as Qwen3.8-Flash-Next's engine does (PP), which the server reads as progress and as
    /// the engine being alive.
    void set_prompt_progress(std::function<void(int64_t read, int64_t total)> f) { prompt_progress_ = std::move(f); }
    /// How forward and prompt pick their tokens (greedy by default): each row's draw counted from its position, so a
    /// seed repeats a run.
    void set_sampler(const strata::kernels::SamplerParams& sp);
    /// A request's constraints on the picks (serve's logit_bias, grammar and penalties, the sampler's chain): a bias
    /// over the vocabulary (empty: none); the grammar's mask for the next pick, (n_vocab + 31) / 32 words (null:
    /// any); the tokens the penalties count, the last ones before the next pick (the sampler's penalty_last_n of them;
    /// none: no penalties).  A window's every row takes the same ones, so a request with them decodes a token a window.
    /// On the last stage, which picks.
    bool set_logit_bias(const std::vector<float>& bias, std::string& err);
    bool set_allowed(const uint32_t* row, std::string& err);
    bool set_history(const int32_t* tokens, int n, std::string& err);
    /// --spec-window min-p's floor for this request (setup's calibration, spec_min_p=); < 0: the engine's own rule
    void set_spec_min_p(double p) { for (PlainModel* st : chain()) st->req_spec_min_p_ = p; }
    /// The PCIe share of the missed experts for this request (setup's calibration, pcie_frac=); < 0: the engine's own
    void set_pcie_frac(double f);
    /// Runs each dense product the prompt takes once, so the products' library has loaded its kernels before the
    /// first prompt (plain_prompt.cpp); nothing when the prompt runs in windows.
    bool prompt_warm(std::string& err);
    /// The states of the prompt tokens [0, n) from position 0: `hidden` takes n rows of n_embd floats, every token's
    /// last block output through the final norm (host), as the prompt computes them before its logits.  A dense model
    /// on one GPU: an encoder (PlainOptions::encoder), or the chat's engine, whose conversation the caller saves and
    /// restores around it.  The state is reset first and left as the prompt leaves it.  False with the reason, also
    /// when the device cannot lend the prompt's buffers.
    bool encode(const int32_t* tokens, int64_t n, float* hidden, std::string& err);

    /// A streamed encoder's (PlainOptions::stream_blocks) block matrices on the device, block by block: begin(l) gives
    /// block l's region for work about to be queued on stream(), end(l) says that work is queued.
    struct BlockSource {
        virtual ~BlockSource() = default;
        virtual const uint8_t* begin(int l) = 0;
        virtual void end(int l) = 0;
    };
    struct HostSpan {
        const void* host = nullptr;   ///< in the mapped file
        uint64_t bytes = 0;
    };
    /// Block l's matrices in the mapped file, in the order set_block_source's offsets take them.
    std::vector<HostSpan> block_spans(int64_t l) const;
    /// Where the blocks' matrices are while the prompt runs: block l's region from `src`, its spans at offsets[l].
    void set_block_source(BlockSource* src, std::vector<std::vector<uint64_t>> offsets);
    /// Queues on stream() what a block's region needs each time its matrices are copied there (the engine's row
    /// layouts of Q6_K and Q8_0, as upload makes them).
    void prepare_block(int64_t l, uint8_t* region);
    strata::gpu::Stream stream() const { return cs_; }
    /// Whether every MoE layer's experts take the int8 products on this GPU (iq_mmq; STRATA_PLAIN_MMQ=0: no).
    bool prompt_int8() const;

    /// Drafts a window can check (0: no speculative decoding): spec - 1 when the model has its MTP layer.
    int drafts() const { return tail().has_mtp_ ? tail().n_snap_ : 0; }
    /// After prompt() (or the windows of a prompt): `first`, the token picked after it, at position `pos`, the next
    /// window's first token; the MTP layer reads it and drafts the tokens after it.
    bool spec_start(int32_t first, int64_t pos, std::string& err);
    /// One window over the pending token and its drafts: `emitted` gets the accepted drafts and the token after them
    /// (`n_emit` of them, at most `limit`), cut after the first token in `stop`; with `logits`, their rows.  The state
    /// then holds every emitted token but the last, which is pending, and the next drafts are made.
    /// `follow` (--spec-follow): the tokens the window must emit instead of its own picks, from the one after the
    /// pending token on (at least `limit`): the drafts that match them are accepted.
    bool spec_step(const std::vector<int32_t>& stop, int64_t limit, int32_t* emitted, int& n_emit, float* logits,
                   std::string& err, const int32_t* follow = nullptr);
    /// The windows and the drafts they checked and accepted since the load, for the report.
    int64_t spec_windows() const { return tail().spec_windows_; }
    int64_t spec_drafted() const { return tail().spec_drafted_; }
    int64_t spec_accepted() const { return tail().spec_accepted_; }
    /// The time spent in the windows that checked drafts, and in making the drafts (ms, since the load).
    double spec_ms_check() const { return tail().spec_ms_check_; }
    double spec_ms_draft() const { return tail().spec_ms_draft_; }

    /// The VRAM tier, for the report: slots, slots holding an expert, and the routed entries it served so far (every
    /// GPU's, with a layer split).
    int64_t slots() const;
    int64_t resident() const;
    /// For the engine's INFO line (generate.cpp's fields): the experts' RAM ("arena": all of them, "resident": the
    /// copy of those no VRAM tier holds, "none": read in place) and its bytes, the tier's bytes
    const char* experts_ram() const;
    uint64_t experts_ram_bytes() const;
    uint64_t tier_bytes() const { return (uint64_t) cache_.mapped_bytes(); }
    /// The CPU pool's worker CPUs (--aux-cpus' plan places the other threads beside them); empty without a pool
    std::vector<int> pool_worker_cpus() const;
    int64_t hits() const;
    int64_t misses() const;
    int64_t swaps() const;
    /// The tiers below VRAM, for the report: the routed entries the GPUs read over PCIe (not in hits or misses), the
    /// experts taken from the RAM copy and from the model's files, and the bytes read from the files.
    int64_t offloaded() const;
    int64_t ram_reads() const;
    int64_t file_reads() const;
    uint64_t file_read_bytes() const;
    /// #533, `VRAM <reserve_mib>` between requests (--vram-elastic): shrinks the VRAM tier until that much VRAM is
    /// free, or grows it back towards its full size when more is free (reserve_mib < 0: the reserve it started with).
    /// `answer` gets the protocol line (VRAM reserve_mib=... expert_slots=...); false with `err` when not elastic.
    bool vram(int64_t reserve_mib, std::string& answer, std::string& err);
    bool vram_elastic() const { return cache_.segmented(); }
    /// --kv-grow: the K/V holds at least `cells` cells (whole steps), the tier giving VRAM up for them; and, when it
    /// holds far more than `cells`, it gives the rest back to the tier.  Between requests; true without --kv-grow.
    bool kv_grow_to(int64_t cells, std::string& err);
    bool kv_trim(int64_t cells, std::string& err);
    bool kv_elastic() const { return kv_cells_ >= 0; }
    /// --control-vector: the loaded vector's tables (kernels/cvec.hpp, cvec_upload) on every stage's GPU too, before
    /// the first window; the blocks then apply it after their FFN (steer)
    bool replicate_cvec(std::string& err);
    /// STRATA_RAM_ADAPT's moves into the RAM copy so far
    int64_t ram_adapts() const { return complement_ != nullptr ? complement_->ram_adapts() : 0; }
    /// ...of them, copied out of a stage buffer
    int64_t ram_staged() const { return complement_ != nullptr ? complement_->ram_staged() : 0; }
    /// The layer split's first blocks after GPU 0's (empty: one GPU).
    std::vector<int64_t> split_points() const;
    /// Writes what this run learned as a profile (rank_learned_profile: the experts in VRAM now, then by the routing
    /// counted since the load, then by the profile it started from), for the next load's `profile`.
    bool save_profile(const std::string& path, std::string& err) const;

private:
    // The layer split: this stage runs blocks [lb_, le_) on the engine's GPU dev_ (-1: the current one).  The first
    // stage (root_) owns the later ones (next_), the experts' RAM tier and the CPU pool, which they share; the public
    // calls go through it.  A stage hands its rows to the next through mapped memory: x (n_embd a row), or GLM's four
    // streams.
    int64_t lb_ = 0, le_ = 0;
    int dev_ = -1;
    std::vector<int> split_devs_;   ///< the root's: the engine's GPUs the stages take, in order
    PlainModel* root_ = this;
    std::vector<std::unique_ptr<PlainModel>> next_;
    bool first() const { return lb_ == 0; }
    bool last() const { return le_ == spec_.n_layer; }
    PlainModel& tail() { return next_.empty() ? *this : *next_.back(); }
    const PlainModel& tail() const { return next_.empty() ? *this : *next_.back(); }
    std::vector<PlainModel*> chain();
    /// A later stage: blocks [lb, le) on GPU `dev`, the experts from root's tiers
    bool load_stage(const PlainOptions& opt, PlainModel* root, int64_t lb, int64_t le, int dev, std::string& err);
    /// The split's points from each GPU's free memory (split_auto)
    bool split_auto_points(std::string& err);
    /// The architecture's part of the load (the blocks of this stage), and after every stage's: the RAM copy of the
    /// experts no GPU holds, each stage's PCIe split, the report
    bool load_arch(std::string& err);
    bool load_qwen(std::string& err);
    void finish_experts();
    void pinned_split();
    /// One stage's part of a window: its blocks over the T rows (from `hand_in`, the previous stage's rows, after the
    /// first), the last stage's picks into h_out_
    bool stage_forward(const int32_t* tokens, int T, int64_t pos, const float* hand_in, std::string& err,
                       bool defer = false);
    /// The recurrent states back to the state after row n (spec_step), every stage's
    bool rollback(int n, std::string& err);
    /// The file's MTP layer is run: by the last stage (has_mtp_); the rest keep the recurrent states' copies for it
    bool model_mtp_ = false;
    std::vector<int64_t> moe_local_;   ///< the MoE layers of this stage's blocks, the MTP layer's last
    int64_t moe_trunk_ = 0;            ///< of them the trunk's: the window's rings
    int64_t hand_w_ = 0;               ///< the hand-off's floats a row
    float *h_in_ = nullptr, *m_in_ = nullptr;       ///< mapped: a later stage's input rows
    float *h_hand_ = nullptr, *m_hand_ = nullptr;   ///< mapped: an earlier stage's output rows
    /// pinned: a layer split's prompt rows in and out, up to prompt_chunk of them (each stage reads them in chunks of
    /// its own, as long as its tier lends the buffers for)
    float* hand_host_ = nullptr;
    float* enc_out_ = nullptr;   ///< encode's rows (host): prompt_chunks writes every row there, normed, no logits
    bool in_ram_ = false;              ///< the experts in pinned RAM (an arena), else read in place
    bool level_zero_ = false;
    uint64_t total_b_ = 0;             ///< the GPU's memory
    int64_t lend_from_ = -1;           ///< the slots the prompt path lends, for the RAM copy (finish_experts)
    uint64_t free_b_ = 0;              ///< the free memory the tier was sized from, for the report

    struct W {   // one 2-D projection in VRAM, in its GGUF blocks (or the engine's row order for Q6_K)
        const void* data = nullptr;
        int type = -1;
        int n_in = 0, n_out = 0;
    };
    struct Block {
        bool recurrent = false;
        int64_t gdn = -1, attn = -1;    ///< index among the GDN / attention blocks
        const float *attn_norm = nullptr, *post_norm = nullptr;
        W qkv, z, alpha, beta, ssm_out;                       // GDN
        const float *conv = nullptr, *dt = nullptr, *a = nullptr, *ssm_norm = nullptr;
        W q, k, v, o;                                         // attention
        const float *q_norm = nullptr, *k_norm = nullptr;
        const float* router = nullptr;                        // F32 [n_embd, n_expert]
        const float* sh_gate_inp = nullptr;                   // F32 [n_embd]
        W sh_gate, sh_up, sh_down;                            // the shared expert, or qwen35's dense FFN
    };
    /// stream_blocks: each block's matrices in the mapped file (W::data set as the block is bound), and their source
    struct Streamed {
        W* w = nullptr;
        HostSpan span;
        uint64_t offset = 0;     ///< in the block's region
    };
    std::vector<std::vector<Streamed>> streamed_;
    std::vector<Streamed>* stream_into_ = nullptr;   ///< load_block's block, while it loads
    BlockSource* block_src_ = nullptr;
    /// Block l's matrices at their source's region (streamed blocks; nothing else), and that work queued.
    void bind_block(int64_t l);
    void unbind_block(int64_t l);

    /// A projection's GGUF tensor into VRAM: its first dimension the input, every other the outputs (a 3-D tensor, the
    /// MLA's per-head wk_b and wv_b, is its heads' matrices one after another).  `f32`: an F32 matrix is taken as it is
    /// (Qwen3.6's GDN alpha and beta: llama.cpp's float product)
    bool upload(const GgufModel& m, const std::string& name, W& w, std::string& err, bool f32 = false);
    /// Block `l`'s weights (blk.<l>.*): its mixer, norms, router and shared expert.
    bool load_block(const GgufModel& m, int64_t l, Block& b, std::string& err);
    /// An F32 vector or matrix, or a BF16 one widened to F32 at load (exact: Qwen3.6's MTP layer keeps its two routers
    /// in BF16).
    bool upload_f32(const GgufModel& m, const std::string& name, uint64_t elements, const float*& out, std::string& err);
    bool open_experts(std::string& err);
    /// qwen35's in place of open_experts: no tiers, the prompt's buffers beside the weights
    bool open_dense(std::string& err);
    bool record(int T, bool mtp, std::string& err);
    /// The window (record_window, or the MTP layer's) captured into `exec`, or its segments into `segs`
    bool capture(int T, bool mtp, strata::gpu::Graph*& exec, std::vector<strata::gpu::Graph*>& segs, std::string& err);
    bool run_window(int T, bool mtp, std::string& err);
    /// The window's inputs (embeddings, steps, positions) for `tokens` at pos .. pos + T - 1 into the mapped staging;
    /// `emb` false: no embeddings (a later stage's window, which reads the rows handed on).
    /// `rows_pos`: row t at rows_pos[t] instead (a batch window's rows)
    bool stage(const int32_t* tokens, int T, int64_t pos, std::string& err, bool emb = true,
               const int64_t* rows_pos = nullptr);
    /// `segs`: where a segmented window's graphs go (null: segs_[T])
    bool record_window(int T, strata::gpu::Stream cs, std::string& err, std::vector<strata::gpu::Graph*>* segs = nullptr);
    /// The control vector after block b's FFN write, on T rows of x (kernels/cvec.hpp), where it covers the layer
    void steer(const Block& b, int64_t T, float* x, strata::gpu::Stream cs) const;
    /// One stage's part of forward_slots
    bool stage_slots(const int32_t* tokens, const int* slot, const int64_t* pos, const strata::kernels::SamplerParams* sp,
                     int T, const float* hand_in, std::string& err);
    /// --batch: a slot's copy of the per-sequence state (the live session's in the members below), this stage's
    struct SlotSession {
        std::vector<float*> gdn_state, gdn_conv;
        std::vector<uint16_t*> k_pool, v_pool, k_scale, v_scale;
        std::vector<int8_t*> k_q, v_q;
        std::vector<uint8_t*> k_q4, v_q4;
    };
    std::vector<SlotSession> bslots_;
    /// --kv-grow: the live session's K/V planes in virtual memory, the bytes a page of cells takes in each, and the
    /// cells mapped (-1: no --kv-grow, every cell allocated)
    std::vector<std::unique_ptr<VmmRange>> kv_vmm_;
    std::vector<uint64_t> kv_vmm_page_bytes_;
    int64_t kv_cells_ = -1;
    bool kv_map(int64_t cells, std::string& err);   // every range's chunks for `cells` cells mapped (zeroed)
    bool alloc_batch_slots(std::string& err);
    /// a batch window's row t is slot row_slot_[t]'s (empty: the live session's rows), while it is recorded or run
    std::vector<int> row_slot_;
    std::map<uint32_t, strata::gpu::Graph*> bexec_;                  ///< per set of slots: the batch window
    std::map<uint32_t, std::vector<strata::gpu::Graph*>> bsegs_;     ///< ... or its segments
    int64_t batch_rows_ = 0;   ///< the batch windows' rows so far (the adaptive rounds' clock in them)
    /// The head over T normed rows `x` into logits_.
    void head_logits(const float* x, int T, strata::gpu::Stream cs);
    /// The MTP layer's head for its last row (hn_ into logits_): the drafts' token subset or the whole head, at 2 bits
    /// where that measured faster on this device (int2_); `pos` is that row's position (device).
    void draft_logits(const int32_t* pos, strata::gpu::Stream cs);
    /// int2_ from the model file's head rows, kept where it measures faster (a failure keeps the head as it is).
    void setup_int2();
    /// One block over the window's T rows of x_ (MoE layer `ml` of the tiers, ring `ring` + 1): a segmented window's
    /// graphs go into `segs`.
    bool record_block(const Block& b, int64_t ml, int T, uint32_t& ring, std::vector<strata::gpu::Graph*>& segs,
                      strata::gpu::Stream cs, std::string& err);
    /// The MTP layer over T rows: the embeddings staged in m_emb_, the trunk's rows in mtp_h_; the last row's normed
    /// output into mtp_h_'s first row (the next draft step's input) and its logits.
    bool record_mtp(int T, strata::gpu::Stream cs, std::string& err);
    /// Launches a window's graph (or its segments) and answers its `rings` rings, MoE layers layers[0 ..].
    bool launch_serve(strata::gpu::Graph* g, std::vector<strata::gpu::Graph*>& segs, int64_t rings,
                      const int64_t* layers, int T, std::string& err);
    /// The MTP layer over T rows at positions pos .. pos + T - 1 (tokens[t] the token at pos + t, mtp_h_'s row t the
    /// trunk's row before it): the greedy draft after the last.
    bool mtp_run(int T, const int32_t* tokens, int64_t pos, int32_t* draft, std::string& err);
    /// The chain after mtp_run's first draft: drafts_ up to `want` from position `pos` (the first draft's).
    bool mtp_chain(int want, int64_t pos, std::string& err);
    /// `max_swaps` > 0: this round moves up to that many, past adapt_swaps and the copy budget (after a prompt).
    /// `max_evict` >= 0: of them at most that many evict an expert (the rest fill free slots).
    bool adapt(std::string& err, int64_t max_swaps = 0, int64_t max_evict = -1);
    /// The adaptive tier's copies in flight: their experts become resident once the copies are done (`wait`: wait
    /// for them; else only if they are).
    bool settle_swaps(bool wait, std::string& err);
    /// STRATA_RAM_ADAPT=<n> (opt-in, one GPU, the experts outside the VRAM tier in RAM): after a window, experts the
    /// conversation keeps routing to take the RAM places of unused ones, n of them read from the file (generate.cpp's
    /// rule, ram_adapt_round).
    void ram_adapt();
    bool fill_slot(int64_t layer, int32_t expert, std::string& err);
    /// The prompt path's buffers for chunks of `T` tokens, carved by prompt_carve: the pinned host block allocated at
    /// load for the longest chunk, the device block borrowed from the tier for each prompt (plain_prompt.cpp uses them).
    struct PromptBufs {
        static constexpr int kGroup = 8;   ///< experts dequantized and multiplied in one grouped product
        int64_t T = 0, gu_elems = 0, d_elems = 0, gemm_elems = 0;
        bool int8 = false;                 ///< every MoE layer's experts through the int8 products (no FP16 buffers)
        float *x = nullptr, *hn = nullptr, *mix = nullptr;
        uint16_t* x16 = nullptr;           ///< T rows of the widest projection input, FP16
        float *qkv = nullptr, *z = nullptr, *ab = nullptr, *gate = nullptr, *beta = nullptr, *h = nullptr, *y = nullptr;
        uint16_t* y16 = nullptr;
        float *qfull = nullptr, *qcur = nullptr, *kraw = nullptr, *kcur = nullptr, *vcur = nullptr, *attn = nullptr,
              *attn32 = nullptr;
        int32_t *steps = nullptr, *pos_q = nullptr, *pos_k = nullptr;
        float *rlog = nullptr, *w = nullptr, *shared = nullptr, *sh_g = nullptr, *sh_gate = nullptr, *sh_up = nullptr;
        int32_t* ids = nullptr;
        uint16_t* xs16 = nullptr;          ///< the routed rows grouped by expert, FP16 [T * k, n_embd]
        float *gu = nullptr, *hh = nullptr, *dm = nullptr, *parts = nullptr, *moe_out = nullptr;
        uint16_t* h16 = nullptr;
        int32_t *src = nullptr, *slot = nullptr, *bounds = nullptr;
        int32_t* entry = nullptr;          ///< grouped row -> routing entry: where the int8 down product writes
        uint16_t *dq_gu = nullptr, *dq_d = nullptr;   ///< kGroup experts' FP16 weights
        uint8_t* stage = nullptr;          ///< 2 kGroup expert blobs copied from RAM (two halves)
        uint8_t* h_stage = nullptr;        ///< pinned host: the same two halves, a source's assembled blobs bounced
                                           ///< through them (plain_prompt.cpp)
        uint8_t *xq = nullptr, *hq = nullptr;   ///< the int8 products' activations: the routed rows, the SwiGLU's
        uint8_t* xq2 = nullptr;            ///< the routed rows for the up product of a layer whose gate and up
                                           ///< differ in the blocks' repetition (iq_mmq_type_dup); else null
        int32_t* identity = nullptr;       ///< row i is row i (the int8 products' output map)
        const uint8_t** wtab = nullptr;    ///< a layer's experts' blobs, for the int8 products
        int32_t* rbounds = nullptr;        ///< each group's bounds from its first row (the down product's)
        int32_t* dense_bounds = nullptr;   ///< {0, T}: a dense product as one int8 "expert"
        int64_t cpu_cap = 0;               ///< routed rows the CPU's share of a layer may take
        float* cpu = nullptr;              ///< the CPU's rows, sent back to be scattered into parts
        int32_t* cpu_row = nullptr;        ///< the routing entry each of them belongs to
        uint16_t* gemm_scratch = nullptr;  ///< Gemm::native's dequantized weight
        // pinned host
        float* emb = nullptr;              ///< T rows of n_embd
        int32_t *h_steps = nullptr, *h_pos = nullptr, *h_ids = nullptr, *h_src = nullptr, *h_slot = nullptr,
                *h_bounds = nullptr, *h_entry = nullptr;
        const uint8_t** h_wtab = nullptr;
        float *h_hn = nullptr, *h_cpu = nullptr;   ///< the layer's normed rows, the CPU's output rows
        int32_t* h_cpu_row = nullptr;
        int32_t* h_dense_bounds = nullptr;
        uint8_t* h_nact = nullptr;         ///< the CPU's activations: a token's in the layer's native format
        int32_t* h_rbounds = nullptr;
        // GLM-5.3-Flash's (plain_prompt_glm5.cpp): the four streams, the mHC's mixes, the window's selection steps, the
        // KDA and DSA intermediates, the dense FFN's gate and up
        float *g_hs = nullptr, *g_mixes = nullptr, *g_post = nullptr, *g_comb = nullptr, *g_y = nullptr;
        int32_t* g_steps = nullptr;
        float *g_q = nullptr, *g_k = nullptr, *g_v = nullptr, *g_qc = nullptr, *g_kc = nullptr, *g_vc = nullptr,
              *g_fa = nullptr, *g_f = nullptr, *g_gate = nullptr, *g_beta = nullptr, *g_ga = nullptr, *g_g2 = nullptr,
              *g_o = nullptr, *g_oy = nullptr;
        float *g_qa = nullptr, *g_qr = nullptr, *g_qb = nullptr, *g_qbT = nullptr, *g_absT = nullptr, *g_abs = nullptr,
              *g_kv = nullptr, *g_lat = nullptr, *g_latT = nullptr, *g_voT = nullptr, *g_vo = nullptr, *g_iq = nullptr,
              *g_ik = nullptr, *g_ig = nullptr, *g_iw = nullptr, *g_scores = nullptr;
        int32_t* g_ids = nullptr;
        float *g_ffg = nullptr, *g_ffu = nullptr;
    };
    /// Device buffers in pieces of at most max_alloc_bytes() (a chunk's buffers may take more than a device allots at
    /// once: the RX 9060 XT's HIP adapter allots 1 GiB): each piece's bytes and, once allocated, where it lies.
    struct PromptPieces {
        std::vector<uint64_t> bytes;
        std::vector<uint8_t*> at;
    };
    /// The lowest slot of `n` a prompt chunk may borrow from (prefill::lend_pct of them, 128 left).
    int64_t lend_lowest(int64_t n) const;
    /// --prompt-chunk not given: the chunk from the tier, as Qwen3.8-Flash-Next's --prefill auto.
    void auto_prompt_chunk(uint64_t beside);
    /// Lays the buffers for chunks of `T` tokens out from `dev` / `host` (null: only counts), returning the device and
    /// host bytes.  With `pieces` the device's go into pieces instead: sized (pieces->bytes, each within
    /// max_alloc_bytes()) while pieces->at is empty, else laid out within those pieces in turn (a chunk of at most the
    /// T they were sized for fits: every buffer grows with T).
    void prompt_carve(int64_t T, PromptBufs& b, uint8_t* dev, uint8_t* host, uint64_t& dev_bytes,
                      uint64_t& host_bytes, PromptPieces* pieces = nullptr) const;
    /// The pieces for chunks of `T` tokens (prompt_dev_, not yet in allocs_) when they take at most `room` bytes;
    /// false, nothing held, when they do not or an allocation fails.
    bool alloc_prompt_dev(int64_t T, uint64_t room);
    PromptBufs pb_;
    void* prompt_host_ = nullptr;
    /// The prompt's device buffers of their own, for chunks of up to prompt_dev_T_ tokens, when every expert has a
    /// slot and the VRAM beside the tier holds them (else none: prompt_borrow lends the tier's last slots), and a dense
    /// model's always (open_dense; none there: the prompt runs in windows).
    PromptPieces prompt_dev_;
    int64_t prompt_dev_T_ = 0;
    /// The prompt's device buffers, carved from the tier's last slots for one prompt (prompt_borrow) and given back
    /// after it (prompt_return), with the experts that were in them put back: the tier keeps that memory between
    /// prompts.  The slots go to the layers in turn (slot s is layer s % L's), so the last ones are every layer's.
    bool prompt_borrow(int64_t T, std::string& err);
    /// The chunks from c_begin before c_end (a layer split: one at a time, each through every stage), the rows of a
    /// later stage's from `hand_in`
    bool prompt_chunks(const int32_t* tokens, int64_t n, int64_t pos, int32_t* out, float* logits, std::string& err,
                       int64_t c_begin, int64_t c_end, const float* hand_in);
    bool prompt_chunks_glm5(const int32_t* tokens, int64_t n, int64_t pos, int32_t* out, float* logits,
                            std::string& err, int64_t c_begin, int64_t c_end, const float* hand_in);
    /// prompt() of one stage; with a split, the chunks' agreed length first (prompt_split)
    bool prompt_one(const int32_t* tokens, int64_t n, int64_t pos, int32_t* out, float* logits, std::string& err);
    bool prompt_split(const int32_t* tokens, int64_t n, int64_t pos, int32_t* out, float* logits, std::string& err);
    bool prompt_warm_products(std::string& err);
    /// W's dense products can go in int8: the int8 products are this GPU's (prefill::mmq::chosen, not turned off by
    /// STRATA_PLAIN_DENSE_MMQ=0) and take W's type and shape.
    bool dense_int8_ok(const W& w) const;
    /// A dense product y = x W^T of T float rows (ldx a row) in int8 (iq_mmq, as one expert of T rows), into y (ldy a
    /// row, 0: W's n_out); false where dense_int8_ok is not (the caller takes Gemm::native).
    bool prompt_dense_int8(const W& w, const float* x, int64_t ldx, float* y, int64_t T, int64_t ldy);
    /// A dense model's: the rows from which a dense product is faster through Gemm::native (dequantized to FP16, the
    /// products' library) than in int8, from the two timed on the dense FFN's gate at 16 rows and a whole chunk's
    /// (dense_native_rows_); after prompt_warm_products, in its buffers.
    void prompt_dense_calibrate();
    /// Dense products of at least this many rows go through Gemm::native, fewer in int8 (INT64_MAX: always in int8,
    /// as the MoE models' dense products)
    int64_t dense_native_rows_ = INT64_MAX;
    /// A dense model's: Gemm::native's products from the normed rows with their sums in FP16 (measured faster at a
    /// chunk's rows by prompt_dense_calibrate: cuBLAS's HGEMM on GeForce cards; not oneMKL's on the B70)
    bool dense_half_sums_ = false;
    /// The chunk's routed experts of MoE layer `ml`: pb_.hn its T normed rows (and pb_.x16 their FP16 copy for the FP16
    /// path), pb_.ids their routes; writes pb_.parts in routing order, the routes into `routes` (null: not kept).
    /// `mark(stage)` is the profile's (1 the grouping, 2 the experts).
    struct PromptRouted {
        int64_t cpu_n = 0, cpu_rows = 0, miss_n = 0;   ///< the CPU's experts and rows, the experts not in VRAM
    };
    bool prompt_routed(int64_t ml, int64_t T, bool cpu_share, int32_t* routes, const std::function<void(int)>& mark,
                       PromptRouted& st, std::string& err);
    bool prompt_cpu_experts(int64_t ml, const std::vector<int32_t>& experts, const std::vector<int32_t>& count,
                            const std::vector<int32_t>& start, std::string& err);
    bool prompt_return(std::string& err);
    int64_t borrow_first_ = -1;
    bool borrow_short_ = false;     ///< prompt_borrow: the tier lends too little for the shortest chunk
    std::vector<std::pair<int32_t, int32_t>> borrowed_res_;   // (layer, expert) the borrowed slots held
    struct PromptGemm;   // plain_prompt.cpp: the products' state over pb_.gemm_scratch
    static void prompt_gemm_free(PromptGemm* g);
    PromptGemm& prompt_gemm();
    std::unique_ptr<PromptGemm, void (*)(PromptGemm*)> prompt_gemm_{nullptr, nullptr};
    static void pool_fn(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                        int64_t layer);

    // GLM-5.3-Flash (glm5-next), plain_glm5.cpp
    struct GBlock {
        bool kda = false;
        int64_t idx = -1;                     ///< among the KDA blocks, or among the DSA blocks
        W hc_attn_fn, hc_ffn_fn;              ///< the mHC mixes (Q8_0 [4 n_embd, 24])
        const float *hc_attn_base = nullptr, *hc_attn_scale = nullptr, *hc_ffn_base = nullptr, *hc_ffn_scale = nullptr;
        const float *attn_norm = nullptr, *ffn_norm = nullptr;
        W q, k, v, f_a, f_b, beta, g_a, g_b;  // KDA
        const float *conv_q = nullptr, *conv_k = nullptr, *conv_v = nullptr, *dt = nullptr, *a = nullptr,
                    *o_norm = nullptr;
        W q_a, q_b, kv_a, k_b, v_b;           // DSA: the MLA (k_b, v_b per head)
        const float *q_a_norm = nullptr, *kv_a_norm = nullptr;
        W idx_q_b, idx_k, idx_gate;           // DSA: the indexer
        const float *idx_k_norm = nullptr, *idx_k_norm_b = nullptr, *idx_proj = nullptr, *idx_ape = nullptr;
        W o;                                  // either mixer's output projection
        W ffn_gate, ffn_up, ffn_down;         ///< a leading layer's dense FFN, else the shared expert
        const float *router = nullptr, *router_b = nullptr;   // F32 [n_embd, n_expert], [n_expert]
        float clamp = 0.0f;                   ///< the dense FFN's / shared expert's SwiGLU limit
    };
    bool glm_ = false;
    std::function<void(int64_t, int64_t)> prompt_progress_;   ///< set_prompt_progress (the root's only)
    std::vector<GBlock> gblocks_;
    GBlock gmtp_;                             ///< GLM's MTP layer (has_mtp_): a DSA block of one stream
    bool load_glm5(std::string& err);
    /// `mtp`: the MTP layer's block, without the mHC mixes
    bool load_gblock(const GgufModel& m, int64_t l, GBlock& b, std::string& err, bool mtp = false);
    bool reset_glm5(std::string& err);
    bool record_window_glm5(int T, strata::gpu::Stream cs, std::string& err);
    bool record_gblock(const GBlock& b, int64_t l, int T, uint32_t& ring, std::vector<strata::gpu::Graph*>& segs,
                       strata::gpu::Stream cs, std::string& err);
    /// A DSA block's mixer over T rows: hn_ (its q8_1 in g_xq_) into mix_
    void record_dsa(const GBlock& b, int T, strata::gpu::Stream cs);
    /// A MoE block's FFN over T rows from hn_ (its q8_1 in g_xq_): the routed experts of MoE layer `ml` and the shared
    /// expert into moe_out_
    bool record_moe(const GBlock& b, int64_t ml, int T, uint32_t& ring, std::vector<strata::gpu::Graph*>& segs,
                    strata::gpu::Stream cs, std::string& err);
    /// record_mtp for GLM-5.3-Flash: eh_proj, the DSA block, the MoE, the shared head's norm
    bool record_mtp_glm5(int T, strata::gpu::Stream cs, std::string& err);
    /// x [T][n_in] through w into y [T][n_out] (q8_1 of x into g_xq_ first unless `quantized`)
    void gproj(const W& w, const float* x, float* y, int T, bool quantized, strata::gpu::Stream cs);
    // GLM's per-sequence state: per KDA block its recurrence (H S S) and its three convolutions' last inputs (3 C 3);
    // per DSA block the indexer's key | gate per cell, its pooled keys, the MLA latent per cell (FP16)
    std::vector<float*> g_idx_cache_, g_pooled_;
    std::vector<uint16_t*> g_latent_;
    int64_t g_max_blocks_ = 0, g_cap_ = 0;
    // GLM's window buffers (device)
    float *g_hs_ = nullptr, *g_mixes_ = nullptr, *g_post_ = nullptr, *g_comb_ = nullptr, *g_y_ = nullptr;
    uint8_t* g_xq_ = nullptr;
    float *g_q_ = nullptr, *g_k_ = nullptr, *g_v_ = nullptr, *g_qc_ = nullptr, *g_kc_ = nullptr, *g_vc_ = nullptr,
          *g_fa_ = nullptr, *g_f_ = nullptr, *g_gate_ = nullptr, *g_beta_ = nullptr, *g_ga_ = nullptr, *g_g2_ = nullptr,
          *g_o_ = nullptr, *g_oy_ = nullptr;
    float *g_qa_ = nullptr, *g_qr_ = nullptr, *g_qb_ = nullptr, *g_qbT_ = nullptr, *g_absT_ = nullptr, *g_abs_ = nullptr,
          *g_kv_ = nullptr, *g_lat_ = nullptr, *g_latT_ = nullptr, *g_voT_ = nullptr, *g_vo_ = nullptr;
    float *g_iq_ = nullptr, *g_ik_ = nullptr, *g_ig_ = nullptr, *g_iw_ = nullptr, *g_scores_ = nullptr;
    int32_t *g_ids_ = nullptr, *g_steps_ = nullptr;
    float *g_ffg_ = nullptr, *g_ffu_ = nullptr;
    /// The recurrent states' sizes in bytes (a GDN's or a KDA's; its convolution history), for reset and spec_step.
    size_t state_bytes_ = 0, conv_bytes_ = 0;

    PlainOptions opt_;
    ModelSpec spec_;
    std::unique_ptr<GgufModel> model_;
    int64_t n_vocab_ = 0;
    int64_t nkv_ = 0;                ///< the attention blocks' K/V heads (the same in every one)
    bool has_gate_ = false;          ///< the attention's output gate: [q | gate] per head (qwen35moe)
    bool has_shared_ = false;        ///< the gated shared expert (qwen35moe)
    bool dense_ = false;             ///< a dense FFN in every block and no experts (qwen35)
    /// a dense model's one-row windows with a graph (0) or straight onto the stream (1), or being timed (-1): the
    /// first decode windows in turns, then the faster (STRATA_PLAIN_EAGER=0 / 1 fixes it).  Other models: 0
    int eager_ = 0;
    int eager_trial_ = 0;            ///< one-row windows timed so far
    double eager_ms_[2] = {};        ///< their milliseconds, with a graph and without
    int emb_type_ = -1;
    const uint8_t* emb_ = nullptr;   ///< token_embd in the GGUF mapping (rows dequantized on the host)
    const std::vector<const float*>* emb_rows_ = nullptr;   ///< set_embeddings'
    /// The embedding of token `tok` at position `pos` into `dst` (n_embd floats): a picture's row there, else the
    /// token's; false for a token outside the vocabulary or a row the host cannot dequantize
    bool embed(int32_t tok, int64_t pos, float* dst) const;
    int32_t* mrope_ = nullptr;       ///< --vision: the cells' (t, h, w), device [max_cells][3]
    size_t emb_row_ = 0;
    W head_;
    const float* out_norm_ = nullptr;
    std::vector<Block> blocks_;
    // the MTP layer (has_mtp_): its block, the projection of [norm(embedding) | norm(trunk row)], the two norms, the
    // norm before the head (the trunk's when the file has none)
    bool has_mtp_ = false;
    Block mtp_;
    W mtp_eh_;
    const float *mtp_enorm_ = nullptr, *mtp_hnorm_ = nullptr, *mtp_head_norm_ = nullptr;
    int64_t n_moe_ = 0;   ///< MoE layers of the tiers: the trunk's, then the MTP layer's
    std::vector<void*> allocs_;      ///< every device allocation, freed in the destructor

    // the expert tiers (src_ and pool_ the root's, shared by every stage)
    std::unique_ptr<ExpertSource> src_own_;
    ExpertSource* src_ = nullptr;
    /// The experts read in place, with the ones the VRAM tier does not hold pinned in RAM (pin_complement), or null
    FileExpertSource* complement_ = nullptr;
    PcieSplitTimes split_;
    std::unique_ptr<strata::kernels::cpu::ExpertPool> pool_own_;
    strata::kernels::cpu::ExpertPool* pool_ = nullptr;
    ExpertCache cache_;
    ExpertDispatch disp_;
    MoeWindow win_;
    std::vector<int32_t> res_;                    ///< [moe layer][expert] slot or -1
    std::vector<std::vector<int32_t>> free_;      ///< per moe layer, its free slots
    std::vector<int64_t> slot_layer_;             ///< per slot, the moe layer it holds experts of
    std::vector<double> heat_;                    ///< [moe layer][expert] routed entries since the load
    std::FILE* route_dump_ = nullptr;             ///< --dump-routing's file, or null
    /// `n` tokens' `k` routed experts of MoE layer `layer` into --dump-routing's file
    void dump_routes(int64_t layer, const int32_t* ids, int64_t n, int64_t k);
    struct StatePart { void* dev; size_t bytes; };
    /// The device buffers a snapshot of `tokens` tokens copies (state_bytes, save_state, restore_state)
    /// `slot`: batch slot slot's copy (-1: the live session)
    std::vector<StatePart> state_parts(int64_t tokens, bool recurrent_only = false, int slot = -1) const;
    bool save_parts(int64_t tokens, bool recurrent_only, Snapshot& snap, std::string& err, int slot = -1);
    /// One attention block's K/V buffers in a session (-1: the live one, else a batch slot), as the format has them
    struct KvPtrs {
        uint16_t *k_pool = nullptr, *v_pool = nullptr, *k_scale = nullptr, *v_scale = nullptr;
        int8_t *k_q = nullptr, *v_q = nullptr;
        uint8_t *k_q4 = nullptr, *v_q4 = nullptr;
    };
    KvPtrs kv_ptrs(size_t a, int slot) const;
    /// The buffers of attention block a in session `slot` for `cells` (cell, KV head) pairs and their bytes, in one
    /// order: the state's parts (state_parts), what reset zeroes
    std::vector<StatePart> kv_planes(size_t a, int slot, size_t cells) const;
    /// Allocates them for `cells` pairs in the --kv format (the live session's or a slot's), into allocs_
    bool alloc_kv(KvPtrs& p, size_t cells, bool vmm = false);
    /// `n` cells' K and V (n rows of n_head_kv x head_dim, rotated as the format keeps them) at steps[t * kStepCount]
    /// into session `slot`'s pools of attention block a
    void kv_append_cells(size_t a, int slot, const int32_t* steps, int n, const float* kc, const float* vc,
                         strata::gpu::Stream cs);
    bool restore_parts(const Snapshot& snap, bool recurrent_only, std::string& err);
    std::vector<std::pair<int32_t, int32_t>> prior_;   ///< the profile the tier was filled from
    int64_t n_slots_ = 0;
    int64_t windows_ = 0, swaps_total_ = 0;
    // the decode's PCIe share chosen by measurement (pcie_probe): windows in blocks with it on and off, alternating,
    // their milliseconds and tokens; -1 once chosen (or not to be chosen: no split, or STRATA_PLAIN_DECODE_PCIE given)
    int pcie_probe_ = -1;
    double pcie_probe_ms_[2] = {};
    int64_t pcie_probe_tok_[2] = {};
    /// Before a window: the PCIe share of the window, while it is being chosen.  After it: its time.
    void pcie_probe_before();
    void pcie_probe_after(double ms, int T);
    int64_t tokens_since_adapt_ = 0;   ///< tokens the windows read since the last adaptive round
    int64_t adapt_phase_ = 0;   ///< a layer split's stage: its rounds this many tokens after the root's
    /// The root's: the tokens between rounds (4 x --adapt-every, or core/expert_tier's auto_adapt_period after
    /// kAdaptDecideAt tokens of output when --adapt-every is not given, from the CPU pool's share of the windows' time).
    int64_t adapt_period_ = 0;
    bool adapt_auto_ = false;
    int64_t adapt_rows_ = 0;
    double pool_ms_ = 0, win_ms_ = 0;   ///< this stage's windows: the CPU pool's host time and the whole
    /// The root's period and the stages' phases from it: the rounds in turn, a share of the period apart.
    void set_adapt_period(int64_t period);
    /// The adaptive tier after a window, as Qwen3.8-Flash-Next's (generate.cpp): its first `rows` rows (from `pos0`)
    /// counted, and a round when the text passes a multiple of the period (adapt_period_).
    bool after_rows(int rows, int64_t pos0, std::string& err);
    /// At a window's start: the last round's swaps taken once they are adapt_lag() windows old (waiting for their
    /// copies then), as Qwen3.8-Flash-Next's verify windows.
    bool settle_window(std::string& err);
    int pending_age_ = 0;
    /// After a prompt: the counts from its routing (seed_usage) and the free slots filled (prompt_return's caller).
    bool prompt_settle(std::string& err);
    std::vector<int64_t> trunk_layers_;            ///< moe_local_ less the MTP layer
    std::vector<int64_t> prompt_routed_, prompt_seen_;   ///< the prompts' routings so far, for seed_usage
    uint8_t* fill_tmp_ = nullptr;                 ///< pinned staging for a slot fill

    // per-sequence device state
    std::vector<float*> gdn_state_, gdn_conv_;   ///< per GDN block
    /// The recurrent states after a window's rows 0 .. n_snap_ - 1, per GDN block (gdn_snap_[block][row]), for
    /// spec_step to go back to the last accepted row.
    int n_snap_ = 0;
    std::vector<std::vector<float*>> gdn_snap_, conv_snap_;
    // the drafting: the pending token, its position, the drafts after it; the MTP layer's input rows (device, max_t of
    // n_embd); the counts for the report
    int32_t cur_ = -1;
    int64_t cur_pos_ = 0;
    std::vector<int32_t> drafts_;
    std::vector<float> dprob_;   ///< drafts_[j]'s probability under the MTP layer, for the window rule
    strata::spec::DraftPolicy policy_{strata::spec::DraftPolicy::kMaxT};
    /// Whether the window rule keeps another draft after the n drafts made (opt_.spec_window).
    bool draft_more(int n) const;
    float* mtp_h_ = nullptr;
    // the drafts' head over opt_.draft_vocab's subset (null: the whole head); a draft's pick is its index there
    void* dhead_ = nullptr;
    std::unique_ptr<DraftSubset> subset_;   ///< the token subset dhead_, dvocab_ and n_dvocab_ come from
    std::unique_ptr<DraftHeadInt2> int2_;   ///< the drafts' head at 2 bits (null: as it is)
    /// --kv-ram: a window's gathered latent rows [max_t][g_gather_stride_][L] and a prompt chunk's layer of cells
    uint16_t *g_lat_gather_ = nullptr, *g_lat_stage_ = nullptr;
    int64_t g_gather_stride_ = 0;
    int64_t n_dvocab_ = 0;
    std::vector<int32_t> dvocab_;
    bool mtp_h_set_ = false;   ///< prompt() left the trunk's last row in mtp_h_ (else forward's last row in hn_)
    int last_T_ = 0;           ///< the last forward window's size
    int64_t spec_windows_ = 0, spec_drafted_ = 0, spec_accepted_ = 0;
    double spec_ms_check_ = 0, spec_ms_draft_ = 0;
    std::vector<uint16_t*> k_pool_, v_pool_;      ///< per attention block (FP16; null with kv_int8)
    std::vector<std::pair<int32_t, int32_t>> vram_evicted_;   ///< `vram`'s shrinks: (residency index, slot)
    std::vector<int8_t*> k_q_, v_q_;              ///< ...kv_int8's codes (K8V4's K)
    std::vector<uint8_t*> k_q4_, v_q4_;           ///< ...kv_q4's Q4_0 blocks (K8V4's V)
    std::vector<uint16_t*> k_scale_, v_scale_;    ///< ...and their scales
    /// The attention block a's pools, for the kernels
    strata::kernels::QsaAttnPools kv_pools(size_t a, int slot = -1) const;
    int32_t* page_table_ = nullptr;
    int64_t n_pages_ = 0;

    // the window's buffers (device unless noted)
    strata::gpu::Stream cs_ = nullptr;
    // the adaptive tier's copies (from the pinned RAM copy, beside the windows) and the experts they bring in
    strata::gpu::Stream swap_cs_ = nullptr;
    strata::gpu::Event* swap_ev_ = nullptr;
    struct PendingSwap { int32_t layer, expert, slot; };
    std::vector<PendingSwap> pending_;
    std::vector<strata::gpu::Graph*> exec_;                      ///< per T: the whole window
    std::vector<std::vector<strata::gpu::Graph*>> segs_;         ///< per T: the segments (MoeWindow::segmented)
    std::vector<strata::gpu::Graph*> mexec_;                     ///< per T: the MTP layer
    std::vector<std::vector<strata::gpu::Graph*>> msegs_;
    float *h_emb_ = nullptr, *m_emb_ = nullptr;                  ///< mapped: T rows of n_embd
    int32_t *h_step_ = nullptr, *m_step_ = nullptr;              ///< mapped: T x kStepCount
    int32_t *h_pos_ = nullptr, *m_pos_ = nullptr;                ///< mapped: T x n_head, then T x n_head_kv
    int32_t *h_out_ = nullptr, *m_out_ = nullptr;                ///< mapped: T picked tokens
    float *h_prob_ = nullptr, *m_prob_ = nullptr;                ///< mapped: the last draft's probability
    // the picks' constraints (set_logit_bias, set_allowed, set_history) on the device, and whether each is set
    float* bias_dev_ = nullptr;
    uint32_t* allowed_dev_ = nullptr;   ///< max_t rows of the same mask
    int32_t* hist_dev_ = nullptr;       ///< max_t rows of hist_len_ tokens
    int hist_len_ = 0, hist_cap_ = 0;
    bool bias_on_ = false, allowed_on_ = false;
    double req_spec_min_p_ = -1.0;
    /// the split as the load set it up, for set_pcie_frac to come back to
    PcieSplitTimes* pcie_split_own_ = nullptr;
    int pcie_num_own_ = 0, pcie_probe_own_ = -1;
    bool pcie_req_ = false;
    /// The picks of T rows of logits into out (device), the row t's draw at position pos + t, with the constraints
    void pick(const float* logits, int T, int64_t pos, int32_t* out, strata::gpu::Stream cs) const;
    strata::kernels::SamplerParams sampler_ = [] {
        strata::kernels::SamplerParams p;
        p.greedy = true;
        p.temperature = 0.0f;
        return p;
    }();
    int32_t *step_ = nullptr, *pos_ = nullptr;
    float *x_ = nullptr, *hn_ = nullptr, *mix_ = nullptr;
    uint8_t* xq_ = nullptr;
    float *qkv_ = nullptr, *craw_ = nullptr, *hc_ = nullptr, *z_ = nullptr, *alpha_ = nullptr, *beta_ = nullptr,
          *gate_ = nullptr, *o_ = nullptr, *y_ = nullptr;
    float *qfull_ = nullptr, *qcur_ = nullptr, *kraw_ = nullptr, *kcur_ = nullptr, *vcur_ = nullptr, *attn_ = nullptr,
          *attn32_ = nullptr, *attn_scratch_ = nullptr;
    float *rlog_ = nullptr, *w_ = nullptr, *shared_ = nullptr, *sh_g_ = nullptr, *sh_gate_ = nullptr, *sh_up_ = nullptr,
          *moe_out_ = nullptr, *logits_ = nullptr;
    int32_t* ids_ = nullptr;
    int64_t attn_scratch_floats_ = 0;
};

}  // namespace strata::core
