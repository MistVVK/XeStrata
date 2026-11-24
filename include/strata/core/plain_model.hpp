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
#include "strata/core/expert_tier.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/graph.hpp"
#include "strata/core/moe_window.hpp"
#include "strata/core/vmm.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/spec/draft_policy.hpp"
#include "strata/spec/guess_gate.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace strata {
class GgufModel;
}

namespace strata::core {


class RemoteExperts;
class PlainCpu;

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
    /// --kv tcq4 / tcq5 (4, 5): K and V trellis-coded at 4 or 5 bits a value after the signed 256-point Hadamard
    /// rotation (kv_tcq.hpp, the query rotated the same way and the output back): 132 and 164 B a head's row, against
    /// Q4_0's 144 and INT8's 264.  head_dim 256 only; not GLM-5.3-Flash's caches.
    int kv_tcq = 0;
    /// --kv not given: a dense model whose blocks the GPU would not all hold with the FP16 cache takes the first of
    /// INT8, tcq5 and tcq4 with which it holds them (auto_gpu_layers), else tcq4
    bool kv_auto = true;
    /// --kv-ram on|off (GLM-5.3-Flash): the DSA layers' MLA latent cells in host memory instead of VRAM, which goes to
    /// the expert tier: a window copies the cells each token selected into VRAM once (glm5_latent_gather), a prompt
    /// chunk the layer's cells up to it (Project Maya #86's idea, written anew: there the cells live in VRAM slots).
    /// On by default: with 128K cells the B70's tier went 2,150 -> 2,322 slots and decode 3.84 -> 3.94 tok/s at 56 GB
    /// of RAM; the RTX 4070 started at all (it had 161 MiB left for the tier), and at 64K went 86 -> 172 slots, its
    /// prompt 3.3 -> 7.1 tok/s and decode 2.32 -> 2.30 (UD-IQ3_XXS, 1K + 256, 2026-10-26).  The other models: off.
    bool kv_ram = true;
    /// --kv-resident N|auto|off: stream the attention layers' K/V (Qwen3.8-Flash-Next's --kv-resident, the same shape).
    /// The authoritative K/V lives in pinned host memory; VRAM keeps a ring of the last N cells a layer, and the decode
    /// attention reads the older cells from the host copy over PCIe (the shared qsa_dense_attn kernel). A prompt chunk
    /// stages one layer's cells into a borrowed VRAM pool (kv_stage_from_host). N the resident cells a layer; auto (the
    /// default, -1): from 64K up when the RAM holds the experts beside the KV host copy; off (0): every cell in VRAM.
    /// The sliding-window layers keep their own ring; GLM-5.3-Flash's DSA caches stream with --kv-ram, not this.
    int64_t kv_resident = -1;
    /// --vision (qwen35, qwen35moe): pictures among the prompt's tokens (plain_generate's GENI): their rows in place
    /// of the token embeddings (set_embeddings) and every cell's M-RoPE position from a table (set_positions)
    bool vision = false;
    /// An image model's text encoder (encode: MOSS-Image reads Qwen3.5-9B's states): no head, no decode attention
    /// timed; the prompt runs in chunks only.  A dense model (qwen35) on one GPU.
    bool encoder = false;
    /// An encoder's block matrices left in the mapped file (block_spans): a BlockSource puts each block's on the device
    /// as the prompt walks the blocks (strata-image's LayerStream, within its budget)
    bool stream_blocks = false;
    /// A streamed encoder read for its sizes only (block_spans, encoder_buffer_bytes): the blocks listed and their
    /// norms on the device, nothing else made; such a model cannot run.
    bool dry = false;
    /// --vram-elastic (#533; --serve, one GPU with virtual memory): the expert slots in physical segments of
    /// vram_segment_mib that `vram` gives back to the driver and maps again between requests
    bool vram_elastic = false;
    int64_t vram_segment_mib = 512;
    int max_t = 8;                  ///< the largest window
    /// --device cpu: the dense Qwen3.5 forward on the CPU alone, no GPU opened; `cpu_threads` is the thread count (0:
    /// physical cores, subject to the CPU quota)
    bool cpu = false;
    int cpu_threads = 0;
    /// --gpu-layers N (the dense models: qwen35, plamo3): the first N blocks and the head on the GPU, the other blocks
    /// on the CPU (PlainCpu, their K/V and states in the RAM), the rows handed between them each window and prompt
    /// chunk; N >= the blocks: every one on the GPU; 0: the CPU alone (as --device cpu); -1: not given
    int64_t gpu_layers = -1;
    /// --batch N (--serve): conversations decoded together, one row each in a window (plain_serve's batch slots), each
    /// slot its own copy of the per-sequence state; at most max_t, as many as take half the VRAM free before the
    /// expert tier at most.  Not GLM-5.3-Flash's nor with --vision (one table of rotary positions) for now.
    int batch_slots = 0;
    /// --batch-mtp (on by default) / --no-batch-mtp: with the MTP layer, a slot's row is followed by the draft the
    /// layer made for it where the window has room, and the slot moves two tokens when it holds (Qwen3.8-Flash-Next's
    /// --batch-mtp: one draft a slot a window).  Each slot keeps its recurrent states after its first row to go back to.
    bool batch_mtp = true;
    /// --batch-groups G|auto (--serve, a layer split): the slots in G groups pipelined through the stages, each stage on
    /// another group at once (plain_serve, PlainModel::pipe_tick); 0: not given (auto on a layer split, as
    /// Qwen3.8-Flash-Next's: batch_groups.hpp), -1: auto
    int batch_groups = 0;
    /// --pipeline-windows N (a layer split): 1, the windows a prompt is read through pipelined over the stages
    /// (forward_run; their order and rows as serial, so the same state); 2 (two stages, the MTP layer), the decode
    /// too: the first stage computes a guess of the next window while the last checks this one (spec_step_pl);
    /// 0: serial.  -1: 1 on a layer split.
    int pipeline_windows = -1;
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
    /// --adapt-async: a round's copies from the pinned RAM copy run beside the next windows, and their experts are
    /// taken kPlainAdaptLag windows on (Qwen3.8-Flash-Next's --adapt-async 1); false: the round waits for them, the
    /// blocking tier (--adapt-async 0)
    bool adapt_async = true;
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
    /// --expert-cache-device1..3 N: N more experts on each helper GPU (the first ones no stage runs on), which computes
    /// the rows routed to them instead of the CPU (core/remote_experts.hpp, as Qwen3.8-Flash-Next's); filled in the
    /// profile's order, then expert by expert across the layers, placed "stripe" or "layer"
    /// (--expert-cache-remote-placement)
    std::array<int, 3> expert_cache_remote{};
    std::string expert_cache_remote_placement = "stripe";
    /// --split-roles any|helper and --split-skip-if-fits: the second GPU's role in --layer-split auto (a stage of the
    /// split, a helper expert cache, or none), as Qwen3.8-Flash-Next's (core/split_role.hpp)
    bool split_roles_helper = false;
    bool split_skip_if_fits = false;
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
    /// --window-hashes PATH: a line per serial verify window (index, position, size, a hash of its final normed rows,
    /// its argmaxes), the answer-drift fingerprint generate.cpp writes (strata/spec/window_hash.hpp).  Empty: none.
    std::string window_hashes;
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
    /// --suffix-draft N (--serve): drafts from an earlier repeat of the context's last N+ tokens (prompt lookup), a
    /// window of them where they go past the MTP layer's (models without that layer too); --lookup-chain K: up to K
    /// such tokens after the MTP's drafts.  0: off.  The recurrent states' copies for the longer windows are taken then.
    int suffix_draft = 0, lookup_chain = 0;
    int lookup_chain_min = 3;   ///< --lookup-chain-min: the shortest match --lookup-chain extends on
    /// --spec-window: how many of the drafts a window checks.  "all": every draft made (the fixed window of spec
    /// tokens); "auto": the drafts kept while the tokens each is expected to add, at its calibrated acceptance, raise
    /// the window's expected tokens per ms at the measured window costs (strata::spec::DraftPolicy, as Qwen3.8-Flash-
    /// Next's engine); "min-p": the drafts while each is at least spec_min_p likely under the MTP layer.  "all" by
    /// default: auto decoded as fast at spec 3, no faster (Ornith-1.5 Q6_K, 256 tokens: B70 95.5 / 97.7 against 96.5 /
    /// 95.8 tok/s, RTX 4070 80.0 / 84.2 against 79.4 / 84.7; GLM-5.3-Flash at 56 GB of RAM on the B70 4.17 against
    /// 4.13, 2026-10-26).
    std::string spec_window = "all";
    /// --coupled-draft (STRATA_SPEC_COUPLED=1): under sampling the MTP layer samples its drafts with the request's chain
    /// and the uniform of the window row that checks each (coupled_draft.hpp) instead of taking the most likely token
    bool coupled_draft = false;
    double spec_min_p = 0.0;
    /// --draft mtp|dflash2|auto|off: which drafter proposes (the shared option, generate's too): the model file's MTP
    /// layer, a DFlash2 pack of the model's (--draft-model: GLM-5.3-Flash's and Qwen3.6-35B-A3B's, tools/
    /// dflash2_fetch.py), or none; auto (--serve) measures them at the start and keeps the fastest
    /// (strata/spec/draft_probe.hpp).
    std::string draft = "mtp";
    std::string draft_model;   ///< --draft-model: the model's DFlash2 pack
    double draft_conf = 0.0;   ///< --draft-conf: DFlash2's window confidence (0: the engine's default)
    bool spec_auto = false;    ///< --spec auto: the window is one of the start's candidates (spec and 2)
    /// --draft dflash2: windows of this many drafts (the recurrent states' copies taken at the load, before the tier
    /// is sized, as the MTP layer's are); 0: the MTP layer's
    int draft_windows = 0;
    /// --draft dflash2 under --wide-window auto: the 16-row windows' drafts, taken before the tier only where the tier
    /// cannot give up their room later (not --vram-elastic), so the start can measure them; 0: none
    int draft_windows_wide = 0;
    std::string draft_rule = "learn";   ///< --draft-rule: a request without a draft= key (--draft auto)
    /// --serve, --draft-costs FILE: the window costs' table (as generate.cpp's); draft_costs_key: the server's part of
    /// its key on the command line, the whole key (strata::spec::draft_costs_key) once the devices are up
    std::string draft_costs, draft_costs_key;
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
    /// The rotations a model's layers take, from the options and the file (the GPU's and the CPU's forward alike): the
    /// full-attention layers' (`full`, the scaling the options or the file ask for) and the sliding-window ones' (`swa`:
    /// their own base, not scaled).  False with the reason for a scaling that cannot be.
    static bool rope_scalings(const ModelSpec& s, const PlainOptions& opt, strata::kernels::RopeScaling& full,
                              strata::kernels::RopeScaling& swa, std::string& err);
    const ModelSpec& spec() const { return spec_; }
    int64_t vocab() const { return n_vocab_; }
    /// The model carries a recurrent state (GDN layers) or a ring of sliding-window K/V (plamo3): a sequence can be
    /// extended, not cut back to any prefix (can_rewind says which ones it can).
    bool recurrent() const;
    /// Whether the state can go back to position `to` and read on from there as if the tokens before it had just been
    /// read: no recurrent state, and each sliding-window layer's ring still holds the cells the token at `to` sees
    /// (to - window + 1 .. to - 1).  `end` < 0: the live state, at the end of the last run; else a state of `end`
    /// tokens put back from a snapshot (its rings: the window before `end`).
    bool can_rewind(int64_t to, int64_t end = -1) const;
    /// The forward runs on the CPU (--device cpu): no GPU, no drafts, no pictures, no batches.
    bool on_cpu() const { return opt_.cpu; }
    int64_t max_cells() const { return opt_.max_cells; }
    /// The options as loaded (the --kv format auto_gpu_layers chose)
    const PlainOptions& options() const { return opt_; }
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
        /// each part's element size where it is floating point (2 or 4), for --conversation-save-compress; 0: codes
        std::vector<uint8_t> typesize;
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
    /// A batch window: row t is token tokens[t] of slot slot[t] at its position pos[t], out[t] the token after it,
    /// drawn with sp[t] (the row's position its counter).  T <= max_t.  The slots ascend, each once, or (--batch-mtp)
    /// twice at consecutive positions: its token and the MTP layer's draft after it.
    bool forward_slots(const int32_t* tokens, const int* slot, const int64_t* pos,
                       const strata::kernels::SamplerParams* sp, int T, int32_t* out, std::string& err);
    /// --batch-mtp: whether the slots draft (the MTP layer, the batch slots and the option)
    bool batch_drafts() const;
    /// --batch-groups: a group's batch window on its way through the stages: its rows (as forward_slots', one a slot),
    /// the rows the stage it left handed on, and the tokens the last stage drew
    struct GroupWindow {
        std::vector<int32_t> tokens, out;
        std::vector<int> slot;
        std::vector<int64_t> pos;
        std::vector<strata::kernels::SamplerParams> sp;
        std::vector<float> hand;
        int group = -1;
    };
    /// One tick of the pipelined groups: every window in the stages moves one stage on, `enter` (null: none) into the
    /// first, and each stage computes its own beside the others (a thread each, the CPU pool one at a time); `done` is
    /// the window the last stage finished (null: none).  No adaptive round starts while windows are in the stages.
    bool pipe_tick(GroupWindow* enter, GroupWindow*& done, std::string& err);
    bool pipe_empty() const;
    int stages() const { return 1 + (int) next_.size(); }
    /// --batch-mtp: slot b's recurrent states back to what its first row left in the last batch window it had two in
    /// (its draft was not the token after it)
    bool slot_rollback(int b, std::string& err);
    /// --batch-mtp: the MTP layer over the rows the last batch window kept, each slot's run of them in order: row i is
    /// that window's row rows[i] (its trunk row), read with tokens[i] (the token after it) at position pos[i] (its
    /// position + 1), slot[i] ascending; drafts[j] the draft after the j-th slot's last row, drawn with sp[j]'s chain
    /// under --coupled-draft (else the most likely).  T <= max_t.
    bool mtp_slots(const int* slot, const int* rows, const int32_t* tokens, const int64_t* pos,
                   const strata::kernels::SamplerParams* sp, int T, int32_t* drafts, std::string& err);
    /// The bytes save_state would take for `tokens` tokens
    size_t state_bytes(int64_t tokens) const;

    /// Tokens [0, T) at positions pos .. pos + T - 1 (T <= max_t): out[t] is the greedy token after token t, and with
    /// `logits` the T rows of n_vocab logits.  Positions must follow the previous window's.
    /// `defer`: the window's rows are not yet known to be kept (a verify window): spec_step counts the ones it keeps
    /// (after_rows), else every row counts.
    bool forward(const int32_t* tokens, int T, int64_t pos, int32_t* out, float* logits, std::string& err,
                 bool defer = false);
    /// Tokens [0, n) at positions pos .. through windows of max_t, as forward's one after another; out and logits are
    /// the last window's (its T rows).  --pipeline-windows 1 or more on a layer split: the windows pipelined, stage s on
    /// window j while stage s + 1 is on window j - 1 (the same rows in the same order on each stage, so the same
    /// state).  `progress(read)` after each window the last stage finished.
    bool forward_run(const int32_t* tokens, int64_t n, int64_t pos, int32_t* out, float* logits, std::string& err,
                     const std::function<void(int64_t)>& progress = {});
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
    /// none: no penalties).  A window's every row takes the same mask and history, so a request with them decodes a
    /// token a window; the bias is the same for every row, and a spec window takes it.  On the last stage, which picks.
    bool set_logit_bias(const std::vector<float>& bias, std::string& err);
    bool set_allowed(const uint32_t* row, std::string& err);
    bool set_history(const int32_t* tokens, int n, std::string& err);
    /// A BATCH slot's own reply tail for its penalty window and its own logit bias, as Qwen3.8-Flash-Next's verifier
    /// has them (verify.hpp: set_slot_history, set_slot_logit_bias): the batch windows draw each row with its
    /// request's penalties and bias, so a request reads the same tokens in a slot as alone.  The tail is the tokens
    /// the slot's session consumed since its reply began (the sampler's penalty_last_n of them are counted), the
    /// fed-back head not in it yet.  On the last stage, which picks.
    void set_slot_history(int slot, const int32_t* tail, int64_t n);
    bool set_slot_logit_bias(int slot, const std::vector<float>& bias, std::string& err);
    /// A mask for each of a window's T rows, (n_vocab + 31) / 32 words each (null: any).
    bool set_allowed_rows(const uint32_t* rows, int T, std::string& err);
    /// What spec_step calls with each window (win[0] the pending token, then the drafts) before it reads it, to set
    /// the rows' masks (set_allowed_rows): the script guard's (src/core/script_guard.cpp); empty: none.  Not on
    /// --pipeline-windows 2's windows (spec_step_pl), which run with no mask.
    void set_window_masks(std::function<bool(const int32_t* win, int T, std::string& err)> f) {
        tail().window_masks_ = std::move(f);
    }
    /// --spec-window min-p's floor for this request (setup's calibration, spec_min_p=); < 0: the engine's own rule
    void set_spec_min_p(double p) { for (PlainModel* st : chain()) st->req_spec_min_p_ = p; }
    /// --serve: a request's prompt reads the residency table again, so the windows are not where a switch that changes
    /// when a swap is applied can be measured against the prompt's rounding (auto_pick.hpp).
    void set_serve(bool on) { for (PlainModel* st : chain()) st->serve_ = on; }
    /// The draft policy of the MTP's windows (--draft-costs fixes its costs, strata::spec::DraftCosts)
    strata::spec::DraftPolicy& draft_policy() { return policy_; }
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
    /// `final_norm` false: the last block's output as it is, before the final norm (Qwen-Image's text encoder).
    bool encode(const int32_t* tokens, int64_t n, float* hidden, std::string& err, bool final_norm = true);
    /// An image model's text encoder's additions to a block's output (Qwen3-VL's DeepStack: the vision reader's rows
    /// added to the picture's tokens after the first layers): in encode, after block `layer`, `after` may add to the
    /// chunk's rows x [T][n_embd] (rows row0 .. row0 + T of the prompt), queued on `stream`.  The chat's engine sets
    /// none.
    struct BlockHook {
        virtual ~BlockHook() = default;
        virtual void after(int64_t layer, float* x, int64_t row0, int64_t T, strata::gpu::Stream stream) = 0;
    };
    void set_block_hook(BlockHook* hook) { block_hook_ = hook; }
    /// An encoder's prompt buffers (PlainOptions::encoder): made by encode for the prompt's length, kept for the next
    /// prompt as long, freed here (the image engine gives their VRAM to the DiT between prompts).  The bytes they take
    /// for a prompt of T tokens.
    void encoder_release();
    uint64_t encoder_buffer_bytes(int64_t T);
    /// An image model's text encoder's adapters (TextEncoder): after each of a dense model's block products in encode,
    /// `after` adds what the adapters change from the product's FP16 input x16 to its output y (row stride ldy); while
    /// one is set the products take their FP16 inputs (not the int8 ones, which leave no FP16 copy).  The chat's
    /// engine sets none.
    struct ProductHook {
        enum Which { kQkv, kZ, kAlpha, kBeta, kSsmOut, kQ, kK, kV, kO, kGate, kUp, kDown };
        virtual ~ProductHook() = default;
        virtual void after(int64_t layer, Which which, const uint16_t* x16, int64_t T, float* y, int64_t ldy) = 0;
    };
    void set_product_hook(ProductHook* hook) { hook_ = hook; }
    /// The rows and columns of layer `layer`'s product `which` (false: the layer has none of that kind).
    bool product_shape(int64_t layer, ProductHook::Which which, int64_t& n_out, int64_t& n_in) const;

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
    /// Whether every MoE layer's experts can take the int8 products on this GPU (iq_mmq; STRATA_PLAIN_MMQ=0: no).
    bool prompt_int8() const;
    /// From how many rows an expert a chunk's MoE layer takes the FP16 path instead of the int8 products (where
    /// prompt_int8): prefill::mmq::int8_rows_below on the first MoE layer's format, as Qwen3.8-Flash-Next's prompt
    /// chooses.  INT64_MAX: int8 at every size (STRATA_PLAIN_MMQ=1, or the timing could not run); 0: FP16 at every size.
    int64_t prompt_int8_below() const;

    /// Drafts a window can check (0: no speculative decoding): spec - 1 when the model has its MTP layer.
    int drafts() const { return tail().has_mtp_ ? tail().n_mtp_ : 0; }
    /// The resolved rope scaling of the full-attention layers (the file's and the options' meet, load_qwen): the
    /// startup summary reads it.  A GLM-5.3-Flash (nope MLA) keeps the identity (its rotation is none).
    const strata::kernels::RopeScaling& rope_full() const { return tail().rope_full_; }
    /// The DFlash2 drafter's taps (the shared mechanism, core/dflash2.hpp): the completed output of these layers,
    /// gathered for every window row into tap_dev() ([max_t][ntap][n_embd]; the plain residual has one stream).
    /// Set before load; empty (the default) allocates nothing and records nothing.
    void set_taps(const std::vector<int>& layers) { tap_req_ = layers; }
    const float* tap_dev() const { return tail().tap_dev_; }
    int n_taps() const { return (int) tail().tap_local_.size(); }
    /// DFlash2 beside the model (the engine's): the head its drafts are read through - the drafts' subset when the
    /// model has one (draft_vocab), else the whole head - on the last stage.
    struct HeadView {
        const void* rows = nullptr;
        int type = -1;          ///< the GGUF / engine row type (30: bf16, read with fp32 rows)
        int64_t n_in = 0, n_rows = 0;
        const int32_t* vocab_dev = nullptr;   ///< row i's token (null: row i is token i)
    };
    HeadView draft_head_view() const;
    /// After load: room for windows of `drafts` drafts (the recurrent states' copies, the extra ones from the VRAM free
    /// now; fewer than the load made: the extra ones freed).  False with `err`, nothing changed, when they do not fit.
    /// The window graphs are captured again.  One GPU only.
    bool set_window_drafts(int drafts, std::string& err);
    /// The VRAM set_window_drafts(drafts) would take now (the copies past those there are; 0 where it takes none)
    uint64_t window_drafts_bytes(int drafts) const;
    /// The window graphs dropped: captured again on their next use (the taps went off, the copies changed).
    void drop_window_graphs();
    /// The prompt's DFlash2 taps (prompt() in chunks and forward_run's windows): the rows at or after `first_pos`,
    /// staged on the device for up to `cap_rows` rows and handed to `on` after each chunk or window (the stream
    /// synchronized).  cap_rows 0 or `on` empty: off, the staging freed.
    bool set_prompt_taps(int64_t first_pos, int64_t cap_rows,
                         std::function<bool(const float* taps, int64_t rows, int64_t pos0, std::string& err)> on,
                         std::string& err);
    /// The taps off after load (DFlash2 released): neither the windows nor the prompt write them.
    void taps_off();
    /// The MTP layer given back after the start's measurement chose DFlash2 or no drafter: its own weights (the
    /// allocations its load made), graphs and the drafts' 2-bit head, and its token subset unless `keep_head`
    /// (DFlash2 reads it).  Its K/V pool and its experts' places stay (a saved state and the tier's layout hold them).
    /// The VRAM given back.
    uint64_t release_mtp(bool keep_head);
    /// The most drafts a window can check (spec_step's `ext`): the recurrent states' copies, else a window less one
    int window_drafts() const {
        const PlainModel& t = tail();
        return recurrent() || t.glm_ ? t.n_snap_ : t.opt_.max_t - 1;
    }
    /// The MTP layer's drafts after the pending token (spec_step checks them next)
    const std::vector<int32_t>& pending_drafts() const { return tail().drafts_; }
    /// After prompt() (or the windows of a prompt): `first`, the token picked after it, at position `pos`, the next
    /// window's first token; the MTP layer reads it and drafts the tokens after it.
    /// `no_mtp`: an outside drafter (DFlash2) drafts this request: the MTP layer makes no draft.
    bool spec_start(int32_t first, int64_t pos, std::string& err, bool no_mtp = false);
    /// The MTP layer's drafts a window from now on, 1 .. the load's (the start's measurement keeps a window)
    void set_mtp_drafts(int k) {
        PlainModel& t = tail();
        if (t.n_mtp_load_ < 0) t.n_mtp_load_ = t.n_mtp_;
        t.n_mtp_ = std::clamp(k, std::min(1, t.n_mtp_load_), t.n_mtp_load_);
    }
    /// The start's measurement of the drafters runs: no routing is counted, no expert moves, no one-time choice is
    /// made on its windows (a start without it makes them on the first request's)
    void set_frozen(bool on) { root_->frozen_ = on; }
    /// A token's embedding row on the host (n_embd floats; DFlash2's anchor and mask rows)
    bool embed_token(int32_t tok, float* dst) const { return embed(tok, -1, dst); }
    /// --pipeline-windows 2: a guessed window the first stage computed for a next window that will not come (the
    /// request ended) is given back, so the state holds the accepted tokens only.  Before the next use of the state.
    bool spec_end(std::string& err);
    /// One window over the pending token and its drafts: `emitted` gets the accepted drafts and the token after them
    /// (`n_emit` of them, at most `limit`), cut after the first token in `stop`; with `logits`, their rows.  The state
    /// then holds every emitted token but the last, which is pending, and the next drafts are made.
    /// `follow` (--spec-follow): the tokens the window must emit instead of its own picks, from the one after the
    /// pending token on (at least `limit`): the drafts that match them are accepted.
    /// `ext` (n_ext tokens, --suffix-draft / --lookup-chain, `match` the lookup's match length): drafts the window
    /// checks instead of the MTP layer's, where they start as those do (or there are none) and the window rule
    /// (Qwen3.8-Flash-Next's DraftPolicy::choose, from the measured window costs and the lookup's acceptance by match
    /// length) expects more tokens a ms from them, as many of them as it takes.
    /// `ext_only`: the window is the pending token and `ext` as given (an outside drafter's, DFlash2's: its window rule
    /// chose them) and the MTP layer makes no draft after it; the committed rows' taps are tap_dev()'s rows 0 ..
    /// n_emit - 1.
    bool spec_step(const std::vector<int32_t>& stop, int64_t limit, int32_t* emitted, int& n_emit, float* logits,
                   std::string& err, const int32_t* follow = nullptr, const int32_t* ext = nullptr, int n_ext = 0,
                   int match = 0, bool ext_only = false);
    /// --window-hashes: this window's fingerprint line (strata/spec/window_hash.hpp), from the last stage's final
    /// normed rows.  Opens the file on the first call; a no-op when --window-hashes was not given.
    bool write_window_hash(int64_t index, int64_t pos, int T, const int32_t* argmax, std::string& err);
    /// The position of the pending token (spec_start's, then each window's).
    int64_t pending_pos() const { return tail().cur_pos_; }
    /// The windows and the drafts they checked and accepted since the load, for the report.
    int64_t spec_windows() const { return tail().spec_windows_; }
    int64_t spec_drafted() const { return tail().spec_drafted_; }
    int64_t spec_accepted() const { return tail().spec_accepted_; }
    /// The time spent in the windows that checked drafts, and in making the drafts (ms, since the load).
    double spec_ms_check() const { return tail().spec_ms_check_; }
    double spec_ms_draft() const { return tail().spec_ms_draft_; }
    /// --pipeline-windows 2: the guessed windows started and held
    const strata::spec::GuessGate& pl_gate() const { return tail().pl_gate_; }

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
    bool vram_elastic() const { return cache_.elastic(); }   // segmented, and its VRAM goes back when unmapped
    /// `ARENA DROP <mib> <file>` (arena_drop_command): the coldest experts of a shared arena, by the profile the tier
    /// was filled from, marked dropped before the server unloads the engine.  False with `err` without an arena.
    bool arena_drop(const std::string& line, std::string& answer, std::string& err);
    /// --kv-grow: the K/V holds at least `cells` cells (whole steps), the tier giving VRAM up for them; and, when it
    /// holds far more than `cells` (or `hold`, the cells a parked conversation may need: kv_grow_hold.hpp), it gives
    /// the rest back to the tier.  Between requests; true without --kv-grow.
    bool kv_grow_to(int64_t cells, std::string& err);
    bool kv_trim(int64_t cells, int64_t hold, std::string& err);
    bool kv_elastic() const { return kv_cells_ >= 0; }
    /// --kv-resident: the trailing cells a streamed attention layer keeps in VRAM (0: not streaming)
    int64_t kv_resident() const { return kv_resident_cells_; }
    /// --control-vector: the loaded vector's tables (kernels/cvec.hpp, cvec_upload) on every stage's GPU too, before
    /// the first window; the blocks then apply it after their FFN (steer)
    bool replicate_cvec(std::string& err);
    /// STRATA_RAM_ADAPT's moves into the RAM copy so far
    int64_t ram_adapts() const { return complement_ != nullptr ? complement_->ram_adapts() : 0; }
    /// ...of them, copied out of a stage buffer
    int64_t ram_staged() const { return complement_ != nullptr ? complement_->ram_staged() : 0; }
    /// The layer split's first blocks after GPU 0's (empty: one GPU).
    std::vector<int64_t> split_points() const;
    /// The attention's shapes the kernels take (heads, head_dim, rotated dims, the K/V cache's cells a page: the layout
    /// a saved conversation's pools are in), the one place they are made.
    strata::kernels::QsaShapes attn_shapes() const;
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
    /// the root's --expert-cache-device1..3: each helper's GPU (-1: none) and its cache, every stage's dispatch's
    std::array<int, 3> helper_dev_{-1, -1, -1};
    std::array<std::unique_ptr<RemoteExperts>, 3> helpers_;
    bool open_helpers(std::string& err);
    /// The free slots of each layer take `want`'s (layer, expert) pairs in order, read in parallel; `filled`: how many
    bool fill_tier(const std::vector<std::pair<int64_t, int32_t>>& want, int64_t& filled, std::string& err);
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
    /// The second GPU's role in --layer-split auto (split_auto_points): a stage, a helper expert cache, or none
    /// (core/split_role.hpp, as generate.cpp).  Needs --expert-profile; without one the split by memory stands.
    bool split_auto_role(std::string& err);
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
    bool enc_norm_ = true;       ///< encode's rows through the final norm (else the last block's output)
    BlockHook* block_hook_ = nullptr;   ///< an encoder's additions after a block (set_block_hook)
    int64_t enc_row0_ = 0;       ///< the chunk's first row, for block_hook_
    ProductHook* hook_ = nullptr;   ///< an encoder's adapters (set_product_hook)
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
    /// Whether w's products read the q8_1 rows the norms and kernels write (not BF16 or F32)
    static bool q8_in(const W& w);
    struct Block {
        bool recurrent = false;
        int64_t gdn = -1, attn = -1;    ///< index among the GDN / attention blocks
        const float *attn_norm = nullptr, *post_norm = nullptr;
        W qkv, z, alpha, beta, ssm_out;                       // GDN
        // alpha and beta widened to F32 for the decode's fused kernel (gdn_ab_multi reads BF16 or F32) where the file
        // has another type; the prompt keeps the file's (its F32 product cost Qwen3.5-9B Q4_K_M's 1K and 32K prompts
        // 3 to 5% on the B70, 2026-11-12).  Unset (data null): the decode reads alpha and beta.
        W alpha_f, beta_f;
        const float *conv = nullptr, *dt = nullptr, *a = nullptr, *ssm_norm = nullptr;
        W q, k, v, o;                                         // attention
        const float *q_norm = nullptr, *k_norm = nullptr;
        /// plamo3: the RMSNorms after the attention and after the FFN (their outputs normed before the residual's add),
        /// the fused [q | k | v] and [gate | up] matrices q, k, v, sh_gate and sh_up are row ranges of, and whether the
        /// layer is a sliding-window one
        const float *attn_post = nullptr, *ffn_post = nullptr;
        W qkv_all, gu_all, beta_alpha_all;
        W hc_attn_fn, hc_ffn_fn;
        const float *hc_attn_bias = nullptr, *hc_ffn_bias = nullptr;
        bool swa = false;
        const float* router = nullptr;                        // F32 [n_embd, n_expert]
        const float* sh_gate_inp = nullptr;                   // F32 [n_embd]
        W sh_gate, sh_up, sh_down;                            // the shared expert, or qwen35's dense FFN
    };
    // QSA indexers are per visit, even when the loop reuses the attention block's ordinary weights.
    struct QsaWeights {
        W q, k;
        const float *q_norm = nullptr, *k_norm = nullptr, *block_norm = nullptr;
    };
    bool sparse_qsa_ = false;
    /// This window's rows all within the indexer's budget, where the selection is every cell: the QSA layers append
    /// their indexer keys and attend densely, without the query, the scores and the top-k (stage_forward; sexec_)
    bool qsa_short_ = false;
    std::vector<QsaWeights> qsa_weights_;
    std::vector<strata::kernels::QsaIndexerBuffers> qsa_index_;
    std::vector<std::vector<float*>> qsa_tail_snap_;
    float *qsa_raw_ = nullptr, *qsa_query_ = nullptr, *qsa_scores_ = nullptr;
    int32_t* qsa_ids_ = nullptr;

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
    /// (Qwen3.6's GDN alpha and beta: llama.cpp's float product), and an F16 one widened to F32 (exact; Unsloth's
    /// Qwen3.5 UD files keep alpha and beta in F16, which no product kernel reads).  `widen`: any type but BF16 and F32
    /// widened to F32 (exact; the GDN's alpha_f and beta_f)
    bool upload(const GgufModel& m, const std::string& name, W& w, std::string& err, bool f32 = false, bool widen = false);
    /// The decode layouts of the matrices upload put on the GPU since the last call (native_mmvq_tune), timed over
    /// all of them: the model's weights, read as the decode reads them, rather than a span allocated for the timing.
    bool tune_uploaded(std::string& err);
    /// Block `l`'s weights (blk.<l>.*): its mixer, norms, router and shared expert.
    bool load_block(const GgufModel& m, int64_t l, Block& b, std::string& err);
    /// An F32 vector or matrix, or a BF16 one widened to F32 at load (exact: Qwen3.6's MTP layer keeps its two routers
    /// in BF16).
    bool upload_f32(const GgufModel& m, const std::string& name, uint64_t elements, const float*& out, std::string& err);
    /// The head's tensor: output.weight, or token_embd.weight where the file ties the two (as llama.cpp reads Qwen3.5's
    /// small models, which have no output.weight)
    const char* head_name() const;
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
    void tap_block(const Block& b, int T, strata::gpu::Stream cs);
    /// set_taps' layers on this stage and the window's tap staging (both loads: load_qwen's and load_glm5's)
    bool setup_taps(std::string& err);
    /// A tapped layer's output rows (R: [T][hc][n_embd]) into the window's taps (`window`) or the prompt's staging
    void tap_rows(int64_t l, const float* R, int hc, int64_t T, int64_t p0, bool window, strata::gpu::Stream cs);
    /// The prompt's staging handed on after a chunk or a window of rows from p0 (prompt_taps_)
    bool prompt_taps_deliver(const float* staging, int64_t T, int64_t p0, strata::gpu::Stream cs, std::string& err);
    /// One stage's part of forward_slots
    bool stage_slots(const int32_t* tokens, const int* slot, const int64_t* pos, const strata::kernels::SamplerParams* sp,
                     int T, const float* hand_in, std::string& err);
    /// --batch: a slot's copy of the per-sequence state (the live session's in the members below), this stage's
    struct SlotSession {
        std::vector<float*> gdn_state, gdn_conv;
        std::vector<float*> gdn_snap, conv_snap;   ///< --batch-mtp: the states after the slot's first row
        std::vector<uint16_t*> k_pool, v_pool, k_scale, v_scale;
        std::vector<int8_t*> k_q, v_q;
        std::vector<uint8_t*> k_q4, v_q4;
        std::vector<uint8_t*> k_tc, v_tc, tc_ring;
    };
    std::vector<SlotSession> bslots_;
    bool bslot_snaps_ = false;   ///< the slots keep the states after their first row (--batch-mtp)
    /// the root's --batch-groups: the window in front of each stage, whether a tick runs (the stages' threads then
    /// take the CPU pool in turn), and that turn
    std::vector<GroupWindow*> pipe_;
    std::atomic<bool> piping_{false};
    std::mutex pool_mu_;
    /// --kv-grow: the live session's K/V planes in virtual memory, the bytes a page of cells takes in each, and the
    /// cells mapped (-1: no --kv-grow, every cell allocated)
    std::vector<std::unique_ptr<VmmRange>> kv_vmm_;
    std::vector<uint64_t> kv_vmm_page_bytes_;
    int64_t kv_cells_ = -1;
    bool kv_map(int64_t cells, std::string& err);   // every range's chunks for `cells` cells mapped (zeroed)
    bool alloc_batch_slots(std::string& err);
    /// a batch window's row t is slot row_slot_[t]'s (empty: the live session's rows), while it is recorded or run
    std::vector<int> row_slot_;
    /// per set of slots (the low 32 bits) and those of them with two rows (the high): the batch window, or its
    /// segments; the MTP layer's batch windows likewise
    std::map<uint64_t, strata::gpu::Graph*> bexec_, mbexec_;
    std::map<uint64_t, std::vector<strata::gpu::Graph*>> bsegs_, mbsegs_;
    /// row_slot_'s row t is the first of its slot's two
    bool first_of_two(int t) const {
        return t + 1 < (int) row_slot_.size() && row_slot_[(size_t) t + 1] == row_slot_[(size_t) t];
    }
    /// The key of the window whose rows are slot[0 .. T): the slots, and those with two rows
    static uint64_t slots_key(const int* slot, int T);
    /// The MTP layer's batch window over row_slot_'s rows: each slot's last row normed and through the head
    bool record_mtp_slots(int T, strata::gpu::Stream cs, std::string& err);
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
    /// The window's graph for every routed expert in VRAM (rexec_ / mrexec_, `mtp` the MTP layer's), recorded as first
    /// needed: where res_window says so, in place of launch_serve; the host counts after the window (res_count).
    bool res_window(int64_t rings, const int64_t* layers, std::string& err);
    bool launch_res(int T, bool mtp, strata::gpu::Graph*& exec, int64_t rings, const int64_t* layers, std::string& err);
    bool all_resident(int64_t rings, const int64_t* layers, std::string& err);
    bool capture_res(int T, bool mtp, strata::gpu::Graph*& exec, std::string& err);
    void res_count();
    /// A MoE layer's ring and its experts in the window's graph (record_block, record_moe)
    void moe_ring(int64_t ml, int T, uint32_t& ring, const float* rlog, strata::gpu::Stream cs);
    bool moe_experts(int64_t ml, int T, uint32_t ring, std::vector<strata::gpu::Graph*>& segs, strata::gpu::Stream cs,
                     std::string& err);
    /// The MTP layer over T rows at positions pos .. pos + T - 1 (tokens[t] the token at pos + t, mtp_h_'s row t the
    /// trunk's row before it): the greedy draft after the last.
    bool mtp_run(int T, const int32_t* tokens, int64_t pos, int32_t* draft, std::string& err);
    /// The chain after mtp_run's first draft: drafts_ up to `want` from position `pos` (the first draft's).
    /// `all`: up to `want` whatever the window rule says (--pipeline-windows 2's guesses read on past a window)
    bool mtp_chain(int want, int64_t pos, std::string& err, bool all = false);
    /// --pipeline-windows 2: spec_step over two stages, the first a window ahead on a guess (see pipeline_windows)
    bool pipelined_spec() const;
    bool spec_step_pl(const std::vector<int32_t>& stop, int64_t limit, int32_t* emitted, int& n_emit, std::string& err);
    /// the first stage's side: before a guess, its last window's rows' copies and the state it left are kept
    /// (pl_save); after the verdict on that window, its state goes back to row n's (n < T - 1) or to what it left
    bool pl_save(std::string& err);
    bool pl_restore(int n, int T, std::string& err);
    /// the chain from position pos1: a window's drafts, and on to the next window's guess where those are likely to be
    /// taken whole (pl_gate_)
    bool pl_chain(int64_t pos1, std::string& err);
    static float pl_theta();
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
        bool int8 = false;                 ///< the MoE layers' experts can take the int8 products (prompt_int8)
        /// a chunk's layer takes them while its experts average fewer rows than this (prompt_int8_below), the FP16
        /// path from there: INT64_MAX, int8 at every size
        int64_t int8_below = INT64_MAX;
        bool f16 = false;                  ///< the FP16 path's buffers are carved (a chunk's layer may take it)
        float *x = nullptr, *hn = nullptr, *mix = nullptr;
        uint16_t* x16 = nullptr;           ///< T rows of the widest projection input, FP16
        float *qkv = nullptr, *z = nullptr, *ab = nullptr, *gate = nullptr, *beta = nullptr, *h = nullptr, *y = nullptr;
        uint16_t* y16 = nullptr;
        float *qfull = nullptr, *qcur = nullptr, *kraw = nullptr, *kcur = nullptr, *vcur = nullptr, *attn = nullptr,
              *attn32 = nullptr;
        int32_t *steps = nullptr, *pos_q = nullptr, *pos_k = nullptr;
        float *qsa_raw = nullptr, *qsa_query = nullptr, *qsa_scores = nullptr;
        int32_t* qsa_ids = nullptr;
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
        /// --kv-resident: one attention layer's whole K/V, staged from the host copy per layer and chunk (the identity
        /// layout, page_table_ its table); the prompt attention reads it, then the next layer overwrites it
        strata::kernels::KvHostPools kv_stage;
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
    int64_t enc_host_T_ = 0;   ///< an encoder's: the chunk prompt_host_ holds (encoder_buffers)
    /// An encoder's prompt buffers for T tokens (one chunk), made again when the last were shorter or freed.
    bool encoder_buffers(int64_t T, std::string& err);
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
    bool prompt_split_reserve();   ///< prompt_dense_calibrate's split products' buffer
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
    PromptShareChoice prompt_share_3k_;
    strata::gpu::Event* prompt_share_events_[2] = {};
    bool prompt_share_pending_ = false, prompt_share_arm_ = false;
    int64_t prompt_share_layer_ = 0, prompt_share_missed_ = 0;
    bool prompt_routed(int64_t ml, int64_t T, bool cpu_share, bool large_share, int64_t max_cpu, int32_t* routes,
                       const std::function<void(int)>& mark, PromptRouted& st, std::string& err);
    bool prompt_routed_tuned(int64_t ml, int64_t T, bool cpu_share, int32_t* routes,
                             const std::function<void(int)>& mark, PromptRouted& st, std::string& err);
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
                        int64_t tb, int64_t layer);

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
    // --device cpu: the forward, and the picks' constraints on the host (set_logit_bias, set_allowed, set_history)
    std::unique_ptr<PlainCpu> cpu_;
    /// --gpu-layers not given, a dense model on one GPU: the blocks its free VRAM holds after the head, their K/V and a
    /// reserve (opt_.gpu_layers set: all of them, N, or 0 for the CPU alone)
    bool auto_gpu_layers(std::string& err);
    /// --kv not given: the FP16 cache unless it leaves blocks off the GPU, then the first of INT8, tcq5 and tcq4 that
    /// holds them all (not bound by the context length).  `fits(row, &bytes)` the blocks a K/V row of `row` bytes
    /// holds, `kv_row(i8, q4, tcq)` the row's bytes.  The dense and the MoE pick share it, the same order.
    template <typename Fits, typename KvRow>
    void auto_kv_format(Fits fits, KvRow kv_row, int64_t nl);
    /// --gpu-layers: the blocks after this GPU's on the CPU (null: none); the head stays here
    std::unique_ptr<PlainCpu> cpu_tail_;
    /// the CPU's blocks over the T rows the GPU's handed on (mapped), then the final norm and the head here
    bool tail_and_head(int T, int64_t pos, std::string& err);
    std::vector<float> cpu_bias_, cpu_logits_;
    std::vector<uint32_t> cpu_allowed_;
    std::vector<int32_t> cpu_hist_;
    int cpu_hist_len_ = 0;
    bool cpu_load(std::string& err);
    bool cpu_run(const int32_t* tokens, int64_t n, int64_t pos, int rows, int32_t* out, float* logits,
                 std::string& err);
    int64_t n_vocab_ = 0;
    int64_t nkv_ = 0;                ///< the attention blocks' K/V heads (the same in every one)
    bool has_gate_ = false;          ///< the attention's output gate: [q | gate] per head (qwen35moe)
    bool has_shared_ = false;        ///< the gated shared expert (qwen35moe)
    bool dense_ = false;             ///< a dense FFN in every block and no experts (qwen35, plamo3)
    bool plamo_ = false;             ///< plamo3: fused products, norms after the mixer and the FFN, sliding windows
    /// the rotation of the full-attention layers (the file's scaling) and of the sliding-window ones (their own base,
    /// no scaling: llama.cpp's freq_base_swa with freq_scale_swa 1)
    strata::kernels::RopeScaling rope_full_, rope_swa_;
    const strata::kernels::RopeScaling& rope_of(const Block& b) const { return b.swa ? rope_swa_ : rope_full_; }
    /// the cells a block's attention reads back from its own: a sliding-window layer's window, else 0 (every cell)
    int64_t window_of(const Block& b) const { return b.swa ? spec_.swa_window : 0; }
    /// rows [r0, r1) of a matrix uploaded whole (a fused product's part): the same blocks, offset by whole rows
    static W row_view(const W& all, int64_t r0, int64_t r1);
    /// a dense model's one-row windows with a graph or straight onto the stream: timed on the first decode windows
    /// (STRATA_PLAIN_EAGER=0 / 1 fixes it).  Other models: with a graph
    StepForm step_form_{"PLAIN_EAGER"};
    int emb_type_ = -1;
    const uint8_t* emb_ = nullptr;   ///< token_embd in the GGUF mapping (rows dequantized on the host)
    const std::vector<const float*>* emb_rows_ = nullptr;   ///< set_embeddings'
    /// The embedding of token `tok` at position `pos` into `dst` (n_embd floats): a picture's row there, else the
    /// token's; false for a token outside the vocabulary or a row the host cannot dequantize
    bool embed(int32_t tok, int64_t pos, float* dst) const;
    /// The embedding rows of `n` tokens asked of the system ahead, all at once, before embed reads them one at a time
    void embed_ahead(const int32_t* tokens, int64_t n) const;
    /// token_embd.weight's place in the mapping and its row format, for `embed` (every model's load)
    bool load_embedding(std::string& err);
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
    std::vector<const W*> tune_wait_;              ///< upload's matrices not yet timed (tune_uploaded)
    std::vector<const void*> weight_at_;           ///< upload's matrices on the GPU, and their bytes
    std::vector<uint64_t> weight_bytes_;
    std::vector<std::vector<float>> widened_;      ///< streamed blocks' F16 matrices upload widened (their spans' host)

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
    std::FILE* hash_file_ = nullptr;              ///< --window-hashes' file, opened on the first window, or null
    /// `n` tokens' `k` routed experts of MoE layer `layer` into --dump-routing's file
    void dump_routes(int64_t layer, const int32_t* ids, int64_t n, int64_t k);
    struct StatePart { void* dev; size_t bytes; int typesize = 0; };   ///< typesize: as Snapshot's
    /// The device buffers a snapshot of `tokens` tokens copies (state_bytes, save_state, restore_state)
    /// `slot`: batch slot slot's copy (-1: the live session)
    std::vector<StatePart> state_parts(int64_t tokens, bool recurrent_only = false, int slot = -1) const;
    bool save_parts(int64_t tokens, bool recurrent_only, Snapshot& snap, std::string& err, int slot = -1);
    /// One attention block's K/V buffers in a session (-1: the live one, else a batch slot), as the format has them
    struct KvPtrs {
        uint16_t *k_pool = nullptr, *v_pool = nullptr, *k_scale = nullptr, *v_scale = nullptr;
        int8_t *k_q = nullptr, *v_q = nullptr;
        uint8_t *k_q4 = nullptr, *v_q4 = nullptr;
        uint8_t *k_tc = nullptr, *v_tc = nullptr;
        uint8_t* tc_ring = nullptr;   ///< --kv tcq's INT8 ring of the cells not coded yet (deferred coding)
    };
    KvPtrs kv_ptrs(size_t a, int slot) const;
    /// The buffers of attention block a in session `slot` for `cells` (cell, KV head) pairs and their bytes, in one
    /// order: the state's parts (state_parts), what reset zeroes
    std::vector<StatePart> kv_planes(size_t a, int slot, size_t cells) const;
    /// the streamed layer a's host copy as state parts (the identity layout, `cells` (cell, KV head) pairs): a
    /// snapshot reads and writes the authoritative host copy, not the VRAM ring
    std::vector<StatePart> kv_host_planes(size_t a, size_t cells) const;
    /// Allocates them for `cells` pairs in the --kv format (the live session's or a slot's), into allocs_
    bool alloc_kv(KvPtrs& p, size_t cells, bool vmm = false);
    /// `n` cells' K and V (n rows of n_head_kv x head_dim, rotated as the format keeps them) at steps[t * kStepCount]
    /// into session `slot`'s pools of attention block a; a streamed layer also writes its host copy, and `stage` (the
    /// prompt path's staging pool) when given; `host_by_dma`: the host copy is left to the caller's DMA from `stage`
    /// (core/kv_host_dma.hpp)
    void kv_append_cells(size_t a, int slot, const int32_t* steps, int n, const float* kc, const float* vc,
                         strata::gpu::Stream cs, const strata::kernels::KvHostPools* stage = nullptr,
                         bool host_by_dma = false);
    bool restore_parts(const Snapshot& snap, bool recurrent_only, std::string& err);
    std::vector<std::pair<int32_t, int32_t>> prior_;   ///< the profile the tier was filled from
    int64_t n_slots_ = 0;
    bool tier_whole_ = false;   ///< every expert has its slot in the tier (the prompt's buffers then beside it)
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
    bool serve_ = false;   ///< --serve: the window probes that change when a swap is applied stay off (set_serve)
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
    int n_mtp_ = 0;   ///< the MTP layer's drafts a window (spec - 1; n_snap_ may be more for --suffix-draft)
    std::vector<int> tap_req_, tap_local_;   ///< set_taps: the DFlash2 drafter's tap layers (this stage's share)
    std::vector<int> tap_slot_;        ///< layer -> its tap index (-1: not tapped)
    float* tap_dev_ = nullptr;         ///< [max_t][ntap][n_embd] window taps (null: none)
    bool taps_on_ = true;              ///< taps_off: nothing is written to them
    float* ptap_dev_ = nullptr;        ///< [ptap_cap_][ntap][n_embd] the prompt's taps (set_prompt_taps)
    int64_t ptap_cap_ = 0, ptap_first_ = 0;
    std::function<bool(const float*, int64_t, int64_t, std::string&)> prompt_taps_;
    int n_snap_load_ = -1;             ///< the copies the model itself needs (the MTP layer's, the lookup's); those after
                                       ///< them are the drafter's, not in allocs_: set_window_drafts may free them
    int n_mtp_load_ = -1;              ///< the MTP layer's drafts the load set (set_mtp_drafts goes no higher)
    size_t mtp_allocs_[2] = {0, 0};    ///< allocs_[0] .. [1]: the MTP layer's own weights (release_mtp)
    bool frozen_ = false;              ///< set_frozen (the root's)
    std::vector<std::vector<float*>> gdn_snap_, conv_snap_;
    /// --pipeline-windows 2, a stage before the last: the rows' copies of the window before a guess and the state that
    /// window left (per GDN block), whether the copies are there, and (the last stage's) the guessed window the first
    /// stage computed: its tokens from position pl_pos_ and the rows it handed on
    std::vector<std::vector<float*>> pl_bak_state_, pl_bak_conv_;
    std::vector<float*> pl_pre_state_, pl_pre_conv_;
    bool pl_bak_valid_ = false, pl_ready_ = false;
    std::vector<int32_t> pl_g_;
    std::vector<float> pl_hand_;
    int64_t pl_pos_ = -1;
    /// the first stage's routing records (ExpertDispatch::route_rec) of the window before a guess and of the guess:
    /// the adaptive tier counts a window's kept rows from its own records, as the serial windows' (after_rows)
    std::vector<int32_t> pl_rec_w_, pl_rec_g_;
    /// the last stage's: whether a guess is worth starting, from the windows' measured costs
    strata::spec::GuessGate pl_gate_{pl_theta()};
    // the drafting: the pending token, its position, the drafts after it; the MTP layer's input rows (device, max_t of
    // n_embd); the counts for the report
    int32_t cur_ = -1;
    int64_t cur_pos_ = 0;
    std::vector<int32_t> drafts_;
    std::vector<float> dprob_;   ///< drafts_[j]'s probability under the MTP layer, for the window rule
    strata::spec::DraftPolicy policy_{strata::kernels::kVerifyNarrowT};   ///< (load: the window, max_t)
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
    std::vector<uint8_t*> k_tc_, v_tc_;           ///< ...kv_tcq's trellis-coded rows
    std::vector<uint8_t*> tc_ring_;               ///< ...and their INT8 ring (kv_tcq.hpp), null when not deferred
    /// --kv tcq: the cells kept as INT8 until their block of 16 is complete, then coded (kv_tcq.hpp); 0 with
    /// STRATA_KV_TCQ_DEFER=0 (each cell coded as it is written), for the comparison
    bool tcq_defer_ = true;
    std::vector<uint16_t*> k_scale_, v_scale_;    ///< ...and their scales
    /// The attention block a's pools, for the kernels
    strata::kernels::QsaAttnPools kv_pools(size_t a, int slot = -1) const;
    int32_t* page_table_ = nullptr;
    int64_t n_pages_ = 0;
    /// The sliding-window layers' K/V in a ring (plamo3): ring_pages_ pages of their own, logical page p held in page
    /// p % ring_pages_ through swa_table_ (0: no ring, those layers' pools as long as the others').  A ring holds the
    /// window and a prompt chunk's cells, so a chunk's writes leave its first query's window in place.
    int64_t ring_pages_ = 0;
    int32_t* swa_table_ = nullptr;
    std::vector<bool> attn_swa_;              ///< per attention block: a sliding-window layer's
    int64_t ring_end_ = 0, ring_from_ = 0;    ///< the position after the last run, and the first the rings still hold
    /// --kv-resident (KV streaming): the attention layers' authoritative K/V in pinned host memory (kv_host_, the
    /// identity layout), VRAM a ring of the last kv_stream_slots_ pages. The decode reads the trailing
    /// kv_resident_cells_ cells from the ring, the older cells from the host copy. 0: not streaming. The
    /// sliding-window layers keep their own ring (attn_swa_), never streamed; the MTP layer's attention streams too.
    int64_t kv_resident_cells_ = 0;
    int64_t kv_stream_slots_ = 0;             ///< the ring's pages (0: not streaming)
    std::vector<strata::kernels::KvHostPools> kv_host_;   ///< per attention block: the host copy (empty: not streamed)
    int32_t* kv_stream_table_ = nullptr;      ///< the ring table, block % kv_stream_slots_ (the streamed layers)
    std::vector<bool> attn_stream_;           ///< per attention block: streamed
    bool kv_streamed(size_t a) const { return kv_stream_slots_ > 0 && a < attn_stream_.size() && attn_stream_[a]; }
    /// resolve --kv-resident (auto/off/N) to the resident cells a layer (0: off), before the K/V is sized; the root
    /// writes the resolved value into opt_.kv_resident so the layer split's stages inherit it
    bool resolve_kv_resident(std::string& err);
    bool kv_resident_fits_vram() const;   ///< the auto rule's VRAM half: the resident K/V beside the own weights
    bool kv_stream_auto_ = false;         ///< auto_gpu_layers planned the ring (the resident K/V pushed blocks off)
    /// the host copy's bytes for this stage's streamed attention layers at the whole context (the auto rule's figure)
    uint64_t kv_host_bytes() const;
    /// allocate attention block a's host copy (the identity layout, the whole context); false on a failed allocation
    bool alloc_kv_host(size_t a, std::string& err);
    const int32_t* table_of(size_t a) const {
        if (kv_streamed(a)) return kv_stream_table_;
        return swa_table_ != nullptr && a < attn_swa_.size() && attn_swa_[a] ? swa_table_ : page_table_;
    }
    bool ringed(size_t a) const { return ring_pages_ > 0 && a < attn_swa_.size() && attn_swa_[a]; }
    /// a run over positions [pos, pos + n) moves the rings' ends
    void ring_advance(int64_t pos, int64_t n);

    // the window's buffers (device unless noted)
    strata::gpu::Stream cs_ = nullptr;
    // the adaptive tier's copies (from the pinned RAM copy, beside the windows) and the experts they bring in
    strata::gpu::Stream swap_cs_ = nullptr;
    strata::gpu::Event* swap_ev_ = nullptr;
    struct PendingSwap { int32_t layer, expert, slot; };
    std::vector<PendingSwap> pending_;
    std::vector<strata::gpu::Graph*> exec_;                      ///< per T: the whole window
    std::vector<std::vector<strata::gpu::Graph*>> segs_;         ///< per T: the segments (MoeWindow::segmented)
    std::vector<strata::gpu::Graph*> sexec_;                     ///< per T: the whole window, qsa_short_'s form
    std::vector<std::vector<strata::gpu::Graph*>> ssegs_;        ///< per T: its segments
    std::vector<strata::gpu::Graph*> mexec_;                     ///< per T: the MTP layer
    std::vector<strata::gpu::Graph*> rexec_, mrexec_;            ///< per T: the two with every routed expert in VRAM
    bool res_rec_ = false;                                       ///< recording rexec_ / mrexec_
    int32_t* res_d_ = nullptr;                                   ///< res_ on the device (null: no resident graphs)
    std::vector<int32_t> res_shadow_;                            ///< res_ as res_d_ holds it
    int32_t *h_rroute_ = nullptr, *m_rroute_ = nullptr;          ///< mapped: a resident window's ids, max_t x k a ring
    std::vector<int64_t> res_rings_;                             ///< the MoE layers of the resident window to count
    int res_t_ = 0;
    std::vector<std::vector<strata::gpu::Graph*>> msegs_;
    float *h_emb_ = nullptr, *m_emb_ = nullptr;                  ///< mapped: T rows of n_embd
    int32_t *h_step_ = nullptr, *m_step_ = nullptr;              ///< mapped: T x kStepCount
    int32_t *h_pos_ = nullptr, *m_pos_ = nullptr;                ///< mapped: T x n_head, then T x n_head_kv
    int32_t *h_out_ = nullptr, *m_out_ = nullptr;                ///< mapped: T picked tokens
    float *h_prob_ = nullptr, *m_prob_ = nullptr;                ///< mapped: the last draft's probability
    // the picks' constraints (set_logit_bias, set_allowed, set_history) on the device, and whether each is set
    float* bias_dev_ = nullptr;
    uint32_t* allowed_dev_ = nullptr;   ///< max_t rows of the same mask
    int32_t* hist_dev_ = nullptr;       ///< max_t rows of hist_len_ tokens; owned separately from allocs_
    int hist_len_ = 0;
    size_t hist_cap_ = 0;
    // the batch slots' own (set_slot_history, set_slot_logit_bias): per slot its reply tail and its bias array
    std::vector<std::vector<int32_t>> slot_tail_;
    std::vector<float*> slot_bias_dev_;   ///< per slot: its bias array on the device (kept for the slot's next one)
    std::vector<char> slot_bias_on_;      ///< ... and whether the slot's current request has a bias
    int32_t* slot_hist_dev_ = nullptr;    ///< the batch window's slots' penalty-history rows (device)
    strata::kernels::SamplerParams* slot_params_dev_ = nullptr;
    size_t slot_params_cap_ = 0;
    size_t slot_hist_cap_ = 0;            ///< ... how many int32 fit; grows to the widest window
    bool bias_on_ = false, allowed_on_ = false;
    std::function<bool(const int32_t*, int, std::string&)> window_masks_;   ///< set_window_masks
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
    float *qkv_ = nullptr, *hc_ = nullptr, *z_ = nullptr, *beta_ = nullptr, *gate_ = nullptr, *y_ = nullptr;
    int32_t* counts_ = nullptr;   ///< 0 .. max_t on the device: a window's rows, as the GDN kernels read their count
    float *qfull_ = nullptr, *qcur_ = nullptr, *kraw_ = nullptr, *kcur_ = nullptr, *vcur_ = nullptr, *attn_ = nullptr,
          *attn32_ = nullptr, *attn_scratch_ = nullptr;
    float *rlog_ = nullptr, *w_ = nullptr, *shared_ = nullptr, *sh_g_ = nullptr, *sh_gate_ = nullptr, *sh_up_ = nullptr,
          *moe_out_ = nullptr, *logits_ = nullptr;
    int32_t* ids_ = nullptr;
    int64_t attn_scratch_floats_ = 0;
};

}  // namespace strata::core
