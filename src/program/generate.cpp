// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/program/generate.cpp - P2.S6: `strata generate`.
//
// THE DRIVER, and the first program in this project that answers a question.  Everything below it is a
// component; this is the thing that composes them into a token:
//
//     embed_row(token)  ->  48 captured layer graphs (the CPU expert pool behind the doorbell)  ->
//     lm_head(R)        ->  sample        ->  embed_row(next)  ->  ...
//
// WHAT IT IS NOT.  There is no tokenizer here.  `pack/full/tokenizer/` and `tools/strata_tokenizer.py` exist,
// and a C++ BPE is Phase 1's deliverable rather than this program's, so the prompt arrives as IDS via
// `--tokens`.  That is not a placeholder: it is exactly what Gate C1 needs, because C1 compares logits against
// llama.cpp on the SAME ids, and a tokenizer on only one side of that comparison is a second variable.
//
// AND IT IS PHASE 2, so hit rate is `h = 0` and the number it prints is slow on purpose
// (`phase-2-correct-engine.md:5-9`).  What it is FOR is the honest tok/s figure and the logit dump.

#include "strata/core/runtime.hpp"
#include "strata/core/device.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/conversation_disk.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/core/conversation_memory.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/mtp.hpp"
#include "strata/core/coupled_draft.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/program/logits_selection.hpp"
#include "strata/program/conv_cache.hpp"
#include "strata/program/message_boundary.hpp"
#include "strata/spec/draft_policy.hpp"
#include "strata/spec/draft_source.hpp"
#include "strata/spec/suffix_drafter.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/core/progress.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <unistd.h>
#include <sys/resource.h>
#include <cerrno>

#include "strata/core/gpu.hpp"

#include <array>
#include <chrono>
#include <algorithm>
#include <memory>
#include <iostream>
#include <limits>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <new>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <vector>

namespace {
// #620 #486: " (N MiB of M MiB VRAM free on this GPU)" for an allocation's failure message
std::string vram_free_note() {
    size_t free_b = 0, total_b = 0;
    if (!strata::gpu::mem_info(&free_b, &total_b)) return {};
    char buf[96];
    std::snprintf(buf, sizeof buf, " (%llu MiB of %llu MiB VRAM free on this GPU)", (unsigned long long) (free_b >> 20),
                  (unsigned long long) (total_b >> 20));
    return buf;
}

// perf-review D-4: the lent slots are refilled with queued copies and one wait; STRATA_REFILL_BLOCKING=1 waits on each
bool refill_blocking() {
    static const bool v = std::getenv("STRATA_REFILL_BLOCKING") != nullptr;
    return v;
}

using Clock = std::chrono::steady_clock;

// The resident RAM mode and the adaptive tier.  A swap copies `in` (held in RAM) into the slot of `out` (held only
// by that slot).  Before the slot is overwritten, `out`'s bytes are copied back from it into an exchange buffer, so
// the CPU computes `out` from RAM while the swap is in flight; when the swap has landed, `commit_exchanges` moves
// them into `in`'s place in RAM.  The RAM copy then again holds exactly the experts no core slot does, and no swap
// reads the file.  Swaps that need no exchange (`out` in the lend region is held in RAM already; or `in` is not) go
// on as before; ones beyond the buffers' room wait for a later round.  Runs on the adaptive tier's thread while the
// GPU commits and drafts: the copies back are on its stream, and waited for before the refills are queued.
//
// With a layer split, `host_res` holds slot numbers of WHICHEVER cache owns the layer, so the copy back reads the
// owning card's cache on that card's stream: `locate(layer)` names them (upstream fe9c10ca: reading a later stage's
// slot number out of the first card's cache put another expert's bytes in RAM, and the answers turned to garbage).
// Upstream #731 (opt-in, STRATA_DISJOINT_ADAPT=1): the adaptive tier leaves an expert a helper GPU holds out of the
// primary's promotion candidates (it would sit in both caches).  Asked live, from the helper's own cache
// (RemoteExperts::holds), so an expert the helper's tier swaps in or out later is followed.
bool helper_holds(const strata::core::ExpertDispatch& d, int64_t layer, int32_t expert) {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_DISJOINT_ADAPT");
        return v != nullptr && std::strtol(v, nullptr, 10) != 0;
    }();
    if (!on) return false;
    for (int r = 0; r < d.remote_count; ++r)
        if (d.remote[r]->holds(layer, expert)) return true;
    return false;
}

struct SwapHome {
    strata::core::ExpertCache* cache;
    strata::gpu::Stream stream;
    int dev;   ///< -1: the first card
};

template <class Swap, class Locate>
bool resident_stage_swaps(strata::core::FileExpertSource& src, const std::vector<int32_t>& host_res, int64_t n_expert,
                          std::vector<Swap>& swaps, Locate locate) {
    if (!src.complement_ready() || swaps.empty()) return true;
    struct Staged { int32_t layer, in, out; int64_t q; };
    std::vector<Staged> staged;
    std::vector<SwapHome> used;   // the streams the copies back went on
    std::vector<Swap> kept;
    kept.reserve(swaps.size());
    for (const Swap& s : swaps) {
        if (!src.has_resident(s.layer, s.in) || src.has_resident(s.layer, s.out)) { kept.push_back(s); continue; }
        const int64_t q = (int64_t) staged.size();
        if (q >= src.exchange_capacity()) continue;
        const int32_t slot = host_res[(size_t) s.layer * (size_t) n_expert + (size_t) s.out];
        if (slot < 0) continue;
        const SwapHome home = locate(s.layer);
        const strata::core::OnDevice on(home.dev);
        if (strata::gpu::copy_async(src.exchange_buffer(q), home.cache->device_slot(slot),
                                    (size_t) strata::kernels::cpu::expert_layout().blob_bytes(s.layer), home.stream) != true)
            return false;
        if (std::find_if(used.begin(), used.end(), [&](const SwapHome& h) { return h.stream == home.stream; }) == used.end())
            used.push_back(home);
        staged.push_back({s.layer, s.in, s.out, q});
        kept.push_back(s);
    }
    if (!staged.empty()) {
        for (const SwapHome& h : used) {
            const strata::core::OnDevice on(h.dev);
            if (strata::gpu::stream_sync(h.stream) != true) return false;
        }
        for (const Staged& x : staged)
            if (!src.stage_exchange(x.layer, x.in, x.out, x.q)) return false;
    }
    swaps.swap(kept);
    return true;
}

struct Options {
    std::string pack = "pack/full";
    std::vector<int64_t> tokens;      // the prompt, PRE-TOKENIZED
    int64_t max_new = 16;
    int64_t max_context = 4096;
    /// The context extension (rope scaling, rope_scaling.hpp; upstream #84): one process configuration, set once before
    /// session_init builds the rope table and captures the kernels (K sits in the cache after RoPE, so one run takes one
    /// scaling).  An explicit flag wins over the model file's rope keys over the defaults; `none` and `1` are explicit.
    std::string rope_scaling;           ///< --rope-scaling none|linear|yarn; empty = the flag is absent
    double rope_scale = 0;              ///< --rope-scale F; 0 = absent (the model file's, else 1 = off)
    double rope_freq_base = 0;          ///< --rope-freq-base N; 0 = the model's (1e7)
    double rope_freq_scale = 0;         ///< --rope-freq-scale F; 0 = 1/--rope-scale
    double yarn_orig_ctx = 0;           ///< --yarn-orig-ctx N; 0 = the model's, else 262144
    double yarn_ext_factor = -1.0;      ///< --yarn-ext-factor F; < 0 = auto (1 for yarn, 0 otherwise)
    double yarn_attn_factor = 1.0;      ///< --yarn-attn-factor F
    double yarn_beta_fast = 32.0;       ///< --yarn-beta-fast F
    double yarn_beta_slow = 1.0;        ///< --yarn-beta-slow F
    bool greedy = true;
    uint64_t seed = 0;
    int top_k = 20;
    float top_p = 0.95f;
    float temperature = 1.0f;
    std::string dump_logits;          // one line of logits per generated position
    int64_t logits_stride = 1;        // storage selection; all prompt tokens remain conditioned
    /// **THE RESIDUAL, SO THE HEAD CAN BE CHECKED WITHOUT THE LAYERS.**
    ///
    /// C1 fails (LEDGER L116) and the pipeline is `embed -> 48 layers -> head`.  Dumping `R` splits it in half:
    /// the head is one norm, two bf16 projections and one 794 MB GEMV, all of which can be recomputed in Python
    /// from the manifest.  If Python agrees with the engine on the same `R`, the head is right and the layers
    /// are wrong; if it disagrees, the head is wrong.  Nothing else in the engine can be split that cheaply.
    std::string ple_gguf;              // the ORIGINAL second GGUF shard: the PLE table is not in the pack
    bool no_ple = false;              // explicit diagnostic ablation; never a normal inference default
    bool stream_token = false;        // R2.6 experiment: ordered work on the session stream
    bool check_logits = false;        // optional full-vocabulary finite scan
    bool gr_fp32_activations = false;  // pinned CUDA single-token BF16 activation contract
    bool gr_native_mmvf = false;       // pinned projection reduction tree as well as FP32 inputs
    bool native_bf16 = false;          // SSM gates, router and indexer projections only
    bool native_bf16_extra = false;    // PLE value and shared expert scalar gate
    bool native_ple_key = false;       // unchanged Q2_0 key and CUDA Q8_1 activations
    bool native_moe_combine = false;   // pinned fused CUDA weighted reduction
    bool native_gdn = false;          // pinned CUDA recurrence and preprocessing
    bool native_flash_attn_short = false; // diagnostic pinned attention, context <=256
    bool native_qsa_indexer = false;  // pinned F16 key cache and F32 pooling
    bool native_qsa = false;          // pinned F32 QSA norms and gate
    bool native_rope = false;         // pinned text-only CUDA rotary arithmetic
    bool native_ple_postops = false;  // pinned PLE postprojection arithmetic
    bool native_router = false;       // pinned fused 512-expert top-10 router
    bool cpu_oracle_q8_0 = false;      // pinned x86 activation scales/codes at both expert stages
    std::string native_head_gguf;      // native output.weight experiment; same model shard as the pack
    std::vector<std::string> native_dense_gguf; // repeat for native GDN/QSA projection shards
    /// Plan v0.3 P1: the whole native arithmetic set as ONE switch (model shard 1). It enables exactly the
    /// combination recorded in bench/results/2026-09-23-attention-ple plus the native indexer, and never the
    /// <=256-token attention adapter. It becomes the default once P0 shows it is not slower.
    std::string native_preset;
    /// Plan v0.3 P2: how the n-gram table is read. Direct (default) = unbuffered SSD reads, table never in RAM.
    std::string ple_io = "direct";
    /// The token embedding from this GGUF instead of --native's (tools/embd_bf16_pack.py: BF16 as shipped)
    std::string embd_gguf;
    int64_t ple_row_cache = 1 << 20;   ///< bounded row cache (rows of 90 B); 0 disables
    int ple_inflight = 64;
    double ple_delay_us = 0;           ///< fault injection: every row read completes no earlier than this
    bool ple_sync_submit = false;      ///< A/B arm: submit reads on the token thread, no I/O worker
    std::string kv = "fp16";           ///< plan v0.3 P7: KV storage, fp16 (default) or int8 (half the VRAM)
    int64_t kv_resident = 0;           ///< KV streaming: resident cells per QSA layer (0: all in VRAM)
    std::string dump_residual;
    /// The head input, `bb.mixed`.  It exists so the head can be SPLIT: steps 1-4 (the per-stream norm, the two
    /// bf16 projections and the stream mean) recompute cheaply in Python, and only the 794 MB GEMV does not.
    std::string dump_mixed;
    /// One residual snapshot per layer per position: `n_layers * hc * n_embd` floats per position, appended in
    /// position order.  This is the C1 BISECTION LADDER - it is what `llama-debug --tensor-filter l_last` prints
    /// for the reference, so the first layer whose `sum` diverges is the layer that holds the bug.  It needs the
    /// captured path (the expert pool only exists there), so it is refused with `--no-capture`.
    std::string dump_layers;
    /// `2 * n_embd + 2 * hc` floats per layer per position: the attention half's block output, the MoE half's,
    /// and the two injection vectors.  It separates `linear_attn_out-<l>` from `ffn_out-<l>`, which the residual
    /// ladder cannot.  **CAPTURED INTO THE LAYER GRAPHS**, so it must be armed before `session_capture`.
    std::string dump_halves;
    /// P0.S8's routing trace, and a prerequisite the Phase 3 plan names explicitly.  One record per layer per
    /// position: `int32 layer, int32 k, k int32 ids, k float weights`.  It is what a hit-rate curve for a
    /// candidate VRAM expert cache is computed from, and it needs no new kernels - the doorbell already
    /// publishes exactly this much to pinned memory.
    std::string dump_routing;
    bool no_capture = false;          // run the layers directly instead of replaying graphs
    bool no_pool = false;             // skip the CPU expert pool: the GPU-only floor
    bool sync_every_layer = false;
    /// Per-stage CUDA-event timings inside the layer halves.  `--no-capture` only: an event recorded inside a
    /// stream capture is silently dropped, so the captured path cannot carry this.
    bool stage_timing = false;
    /// Launch the 48 captured `pre` graphs back to back with no host work between them and report the pure GPU
    /// time per token.  This is the only measurement that separates host-bound from GPU-bound, because the
    /// stage events include every gap where the GPU waited for the host.
    bool graph_only = false;
    bool gpu_only_full = false;   ///< R0.3: pre + post + head, the true per-token GPU floor
    int pool_workers = 0;         ///< R2.2: 0 = "all physical cores minus the host's"; >0 overrides
    int pool_tasks = 0;           ///< Batched CPU expert tasks per phase; 0 keeps the existing policy
    /// R2.2's first half, as an A/B arm.  **ON by default**, because the measurement that justifies it is the
    /// pool's own drain: 33.7 GB/s against 5/6 x 44.14 = 36.8 for five workers, on a machine whose sixth core
    /// is reserved for a host thread that has nothing to do while the drain runs.
    bool no_host_worker = false;
    bool mmap_experts = false;    ///< R2.1: opt OUT of the resident arena, back to MapViewOfFile
    /// The low-RAM mode's resident copy: GiB of RAM for the experts the GPU cache does not hold (0 = none; < 0 =
    /// --resident-experts: as much as the free RAM allows)
    double resident_gib = 0;
    bool resident_cpu_experts = false;
    bool resident_pin = false;
    std::string shared_expert_arena;
    bool coupled_draft = strata::core::coupled_draft_env();
    /// R4: slots of VRAM-resident experts.  **0 = off, and off is the default.**
    /// **THE COMMENT THAT USED TO BE HERE WAS FALSE AND ROUND 328 MEASURED IT.**  It said "the cache has no
    /// consumer yet - `moe_hit_grouped_s2` does not exist - so switching it on costs the fill traffic and
    /// saves nothing".  The kernel exists (`Strata's src/kernels/cuda/s2_expert_grouped.cu`), it is wired at line ~660
    /// via `expert_hit_run`, and switching the cache on **does** move work off the CPU pool: the drain fell
    /// **19.076 -> 10.312 ms/token** at 4096 per-layer slots, for **-2.7 ms/token** end to end.  What was
    /// true is that the ADMISSION POLICY gave every slot to the first position, which is why the earlier
    /// measurement found nothing - see `expert_cache_per_layer`.
    int expert_cache = 0;
    std::array<int, 3> expert_cache_remote{}; ///< CUDA1..3 slots; CUDA0 keeps dense/state/MTP
    std::string expert_cache_remote_placement = "stripe"; ///< stripe experts or assign complete layers to CUDA1..3
    bool expert_cache_cpu_order = false;
    /// **R4.2g.  ROUND 328 MEASURED THAT THE GLOBAL ADMISSION POLICY CANNOT WORK, AND THIS IS THE FIX.**
    /// The default policy hands out slots in arrival order from one counter shared by all 48 layers, so the
    /// first `n_slots` distinct pairs - about 26 LAYERS OF POSITION 0 - take every slot and hits are confined
    /// to them.  Measured at 256 slots: **1781 of 60000 = 2.97%**, against **21.4%** for 8 slots per layer and
    /// **70.4%** for 64, from `Memory/cache_allocation.py` on the same run's routing.  Off by default.
    bool expert_cache_per_layer = false;
    /// R4.2e: a `profile.bin` from `tools/make_profile.py`.  **When given, it decides residency instead of the
    /// compulsory-miss policy**, which is the whole point: a profile ranked by routing frequency over a whole
    /// trace is what the plan's `h = 0.6447` refers to, and compulsory-miss measured 0.4864 because it fills
    /// with whatever the prompt touched FIRST.  Empty means no profile.
    std::string expert_profile;
    /// #477 (--serve, opt-in): where to save what the adaptive tier learned, as a profile `--expert-profile` reads
    /// (the resident experts first, then the routing counted since the start); on QUIT and every
    /// `expert_profile_save_min` minutes between requests.  Empty (the default): nothing is counted or written.
    std::string expert_profile_save;
    double expert_profile_save_min = 10.0;
    /// R4.2d: **ON by default**, because the measurement is unambiguous and the alternative is known-broken.
    /// Without it, 17 of 10,562 layers had the hit work done when the pool returned; with it, 9,190.  The
    /// A/B arm is `--no-hit-poke`.
    bool no_hit_poke = false;
    /// R0.9: capture each layer as THREE graphs and time them from outside the capture, which is the only
    /// valid way to get a per-stage table on the real graph.  Prints and exits; it is a measurement, not a run.
    bool gpu_stages = false;
    bool stats = false;
    bool shared_late = false;          ///< plan v0.3 P3 A/B: shared expert inside post[l] (old order)
    bool keep_canonical = false;       ///< plan v0.3 P1 A/B: load canonical copies of natively served tensors
    bool no_token_graph = false;       ///< plan v0.3 P3 A/B: two graphs per layer instead of one per token
    bool no_fused_gr = false;          ///< plan v0.3 P3 A/B: the six-kernel native gr_read + separate gr_write
    bool no_fast_attn = false;         ///< plan v0.3 P3 A/B: gather + one-block-per-head QSA attention
    bool no_publish_kernel = false;    ///< plan v0.3 P3 A/B: memcpy nodes for the doorbell and QSA step
    bool no_fused_gdn = false;         ///< plan v0.3 P3 A/B: llama.cpp-layout GDN step + separate out norm
    bool no_fast_select = false;       ///< plan v0.3 P7 A/B: FP64 row scores + bit-serial cell top-k
    /// Plan v0.3 P4: `--expert-cache auto` sizes the VRAM tier from what is free after the weights, the session
    /// and the KV state, minus this reserve for the graphs, the hit scratch and the head.
    int vram_reserve_mib = 700;
    bool vram_reserve_given = false;   ///< --vram-reserve-mib on the command line (#496: no smaller automatic reserve)
    /// #533 (opt-in): the expert cache in physical segments (virtual memory), so the serve loop's `VRAM <reserve_mib>`
    /// command can give part of it back to other programs and take it again.  Off: one allocation.
    bool vram_elastic = false;
    int64_t vram_segment_mib = 512;
    /// `--vram-reserve-later-mib N`: the reserve on a layer split's later cards (default: the same as the first).
    /// A card that drives no display needs less than the one the monitors are on.
    int vram_reserve_later_mib = -1;
    /// Plan v0.3 P5: batched prompt processing in chunks of this many tokens (0 = the token path).
    int64_t prefill_chunk = 0;
    /// `--prefill auto`: the largest chunk (up to 32768, not past --max-context) whose buffers the expert cache can
    /// lend.  Every expert a chunk routes to is streamed once per chunk, so a bigger chunk streams fewer bytes per
    /// token (the "ubatch" effect).
    bool prefill_auto = false;
    bool no_split_rows = false;        ///< plan v0.3 P4 A/B: one whole expert per pool thread
    /// Plan v0.3 P5: the prompt path borrows the top expert-cache slots for its buffers and refills them after
    /// the prompt (default); `--no-prefill-borrow` reserves the buffers' VRAM for the whole session instead.
    bool no_prefill_borrow = false;
    /// Plan v0.3 P5 validation: batch only positions [0, P) and run the rest of the prompt through the token path
    /// (teacher-forced), so the logits of positions >= P - which depend on the batched state - can be scored
    /// against the oracle at many positions.  0 = the whole prompt but the last position.
    int64_t prefill_until = 0;
    int prefill_experts = 16;          // --prefill-experts: experts a grouped product of the prompt path takes
    /// Plan v0.3 P6: after every processed position, append the residual after the last layer (hc x n_embd
    /// floats, the MTP draft head's input) to this file.  Token path only.
    std::string dump_final_r;
    /// Plan v0.3 P6: speculative decoding with a verify window of this many tokens (the last accepted token and
    /// spec-1 drafts); 0 = plain decode.  `spec_oracle` drafts from a token file (the expected continuation, for
    /// the exactness test); `spec_corrupt` N > 0 replaces every Nth draft with a wrong token.
    int spec = 0;
    /// --serve runs up to this many sequences at once, one token each per batch window (0 = off; upstream PR #559,
    /// #465).  A request sent as `BGEN <slot> <max_new> ...` reads its prompt and its first token through the usual
    /// path, then continues in slot <slot> of the batch windows (`BT <slot> <id>` lines, then `BDONE <slot> ...`).
    int batch = 0;
    /// With a layer split: the --batch slots in this many groups pipelined through the stages (stage k runs one group
    /// while stage k+1 runs another; upstream PR #559).  1 = every slot in one window, stage after stage.
    int batch_groups = 1;
    /// EXPERIMENTAL, off by default: with --batch-groups over two GPUs, the CPU expert workers in two pools on separate
    /// cores (N0 for the first stage, N1 for the second), each stage's windows served by a host thread of its own, so
    /// the stages' CPU experts run at once.  On the B70 + RTX 4070 PC (DDR, i7-14700) it was slower: the two stages
    /// share the RAM bandwidth the CPU experts are bound by (69 -> 62 tok/s); kept for PCs with more of it.
    int batch_cpu_split[2] = {0, 0};
    std::string spec_oracle;
    int spec_corrupt = 0;
    /// Benchmarks and A/B checks (eddoursul's fork, F19): the run emits this continuation (token ids) instead of
    /// the window's argmax, and a draft is accepted when it matches it, so runs with different speculation
    /// settings process the same text.
    std::string spec_follow;
    /// Per verify window: a line with its size, a hash of its final residual rows and its argmaxes.
    std::string window_hashes;
    /// Plan v0.3 P6: the MTP draft layer's runtime directory (tools/mtp_rt.py); drafts come from it.
    std::string mtp;
    int64_t mtp_window = 32768;   ///< the draft layer attends to the last N cells (0 = every cell)
    /// Plan v0.3 P6: the share (0..1) of each layer's distinct missed experts the GPU reads over PCIe from the
    /// pinned arena while the CPU computes the rest (verify windows).
    double pcie_frac = -1.0;   ///< < 0: the model's default (0.2 direct for the Q2_0 pack, 0.55 DMA for native packs)
    std::string pcie_mode = "auto";   ///< auto | dma | kernel | direct
    /// Plan v0.3 P6: every `adapt_every` rounds, swap up to `adapt_swaps` of the most-routed missing experts into
    /// the VRAM tier in place of the least-routed resident ones (decayed counts).  0 = static residency.
    int adapt_every = 4;
    float adapt_decay = 0.7f;   ///< the usage counts are multiplied by this after each adaptation (--adapt-decay)
    /// Plan v0.3 P6: a draft enters the verify window only while every draft before it (and itself) has at least
    /// this probability under the draft layer; 0 = always --spec-1 drafts.
    double spec_min_p = 0.0;
    /// Stop when the model emits an end-of-turn token (<|endoftext|> 248044, <|im_end|> 248046, or --eos-ids).
    bool stop_eos = false;
    std::vector<int64_t> eos_ids = {248044, 248046};
    bool spec_split = false;   ///< opt-in split verify window (the overlap study: exact, ~7% slower)
    /// --serve, multi-GPU layer split: "K" or "K1,K2,.." (the first layer of each later stage) or "auto" (placed
    /// from each GPU's free VRAM); empty = one GPU
    std::string layer_split;
    /// the later stages' devices "D1,D2,.." (default: the next visible GPUs; "0" with one K: both stages on this
    /// GPU, sharing everything - the bit-exact A/B of the hand-off)
    std::string split_device;
    /// Plan v0.3 P8: stay resident and take requests on stdin (see the --serve block in main).
    bool serve = false;
    /// The vision path: keep a per-cell (t, h, w) rotary position table so --serve can take GENI requests.
    bool vision = false;
    int adapt_swaps = 96;
    /// --serve: how many conversation checkpoints to keep between requests (0 = every request reads its whole
    /// prompt again, the v0.1.2 behaviour).  One is the GDN recurrence of the 36 layers, the QSA indexer tails and
    /// the PLE history (~118 MB of host RAM); the KV cache itself is positional and stays where it is.
    int prompt_cache = 6;
    /// --serve: parked conversations (upstream ccc660b): a host-RAM budget for whole sessions put aside when another
    /// conversation takes the session, at most `slots` of them, and the RAM that must stay free when one is parked
    int64_t conversation_cache_mib = 0;
    int conversation_cache_slots = 4;
    int64_t conversation_cache_min_free_mib = 2560;
    /// --serve: write parked conversations to this folder (when the RAM cache pushes one out, and at QUIT) and
    /// read them back after a restart; empty = off.  Checkpoints per conversation (the deepest, the chain's root,
    /// then the most recently used), the folder's budget, and the age after its last use at which a file goes.
    std::string conversation_save;
    int conversation_save_checkpoints = 2;
    int64_t conversation_save_mib = 16384;
    int64_t conversation_save_hours = 48;
    bool conversation_save_compress = false;   // the floating-point parts through c-blosc2 (built only when found)
    /// --serve SAVE: disk space a session file must leave free where it is written (MiB; 0 = no check)
    int64_t session_min_free_mib = 4096;
    /// --serve: also keep a checkpoint every N freshly read prompt tokens (0 = only at the last turn boundary)
    int64_t prompt_cache_every = 16384;
    bool prompt_cache_tail = false;   // optional extra checkpoint at an existing near-tail chunk boundary
    /// --serve: a prompt read from token 0 is also checkpointed at its first turn boundary - the end of the system
    /// prompt, which every chat of the same client shares - when that is at least N tokens (0 = never)
    int64_t prompt_cache_root = 2048;
    /// --serve: the token that opens a chat turn (<|im_start|>).  The last one in a prompt is where the chat's
    /// history ends and the new assistant turn begins, which is the checkpoint the next request can reuse.
    int64_t turn_token = 248045;
    /// --serve (#458, opt-in): the role token of a short turn the server puts right before the new assistant turn
    /// (the "system" of a trailing reasoning-effort turn).  When the turn before the last <|im_start|> opens with it,
    /// the checkpoint goes in front of that turn instead, so it holds the conversation and not the effort text.
    /// -1 = off (the checkpoint at the last <|im_start|>, as before).
    int64_t tail_role_token = -1;
    /// --serve: a text part of the prompt of at most N tokens (a chat message, the assistant header, a short tool
    /// result) goes through the verify windows, S tokens at a time, instead of the batched prompt path (0 = always
    /// the batched path)
    int64_t short_read = 64;
    /// The suffix drafter (prompt lookup): when the text being written repeats an earlier stretch of the context (code
    /// edits, quoted input, tool-call JSON) by at least this many tokens, the window may be filled with what followed
    /// it there instead of the MTP's drafts, where the MTP's own first guess agrees and the draft policy expects it to
    /// pay (strata/spec/draft_policy.hpp).  On by default; 0 = MTP only.
    int suffix_draft = 3;
    /// The MTP's own window cap (0 = --spec): with --spec 6 --mtp-max-t 4 the long windows come from suffix matches.
    int mtp_max_t = 0;
    /// --lookup-chain K (opt-in, 0 = off): prompt lookup CHAINED after the MTP's proposal.  The suffix drafter matches
    /// the context followed by the MTP's drafts and appends up to K of the tokens that followed that match to the same
    /// verify window (the window grows to at most 8).  Greedy verification keeps the output; its own counts are kept.
    int lookup_chain = 0;
    int lookup_chain_min = 3;   ///< --lookup-chain-min M: the shortest match (in tokens, drafts included) it extends on
    /// --mtp-hnorm stream (opt-in): the draft layer's pre_fc_norm_hidden normalizes each hyper-connection stream on
    /// its own (llama.cpp's qwen4exp MTP graph) instead of one RMS over all four (the default, tools/mtp_probe.py).
    bool mtp_hnorm_stream = false;
    /// A control vector on the residual stream (strata/kernels/cvec.hpp), with llama.cpp's flags: the
    /// `experimental-speed-projection` profile passes `--control-vector-scaled FILE:1.0 --control-vector-layer-range
    /// 4 44 --cvec-mode project --cvec-dir per-layer`.  None by default; --serve switches a loaded one per request.
    std::vector<std::pair<std::string, float>> cvec_files;
    int cvec_first = -1, cvec_last = -1;   ///< llama.cpp's defaults: 1 .. the last layer
    int cvec_mode = 1;                     ///< 0 = project, 1 = add (llama.cpp's default)
    int cvec_single = -1;                  ///< --cvec-dir single:L (project mode): layer L's direction everywhere
};

void usage() {
    std::fprintf(stderr,
                 "strata generate --pack DIR --tokens \"1,2,3\" [options]\n"
                 "\n"
                 "  --pack DIR           the pack directory (default pack/full)\n"
                 "  --tokens LIST        the prompt as comma-separated token IDS (required)\n"
                 "  --tokens-file PATH   pretokenized prompt, commas or whitespace (alternative to --tokens)\n"
                 "  --ple-gguf PATH      the PLE table (with --native: default the shard of --native's model that holds it)\n"
                 "  --no-ple             explicit diagnostic ablation of the PLE layer\n"
                 "  --ple-io direct|mmap|ram  n-gram table reads (plan v0.3 P2). direct (default): unbuffered SSD\n"
                 "                       reads, the table never enters RAM or the file cache; mmap: A/B arm;\n"
                 "                       ram: mmap with the whole table locked in RAM at start\n"
                 "  --embd-gguf PATH     the token embedding from this GGUF instead of --native's (tools/embd_bf16_pack.py:\n"
                 "                       BF16 as the checkpoint ships it; mapped host memory, no VRAM)\n"
                 "  --ple-row-cache N    bounded cache of fetched rows, 90 B each (default 1048576; 0 = off)\n"
                 "  --ple-inflight N     outstanding SSD reads (default 64)\n"
                 "  --ple-delay-us U     fault injection: each row read completes no earlier than U us\n"
                 "  --ple-sync-submit    A/B arm: submit table reads on the token thread (default: an I/O thread)\n"
                 "  --kv fp16|int8       KV storage (plan v0.3 P7): int8 codes + fp16 scale per 64 values, half the\n"
                 "                       VRAM; default fp16 until gate G-C accepts int8\n"
                 "  --kv q4_0            4-bit K/V after a Hadamard rotation (PR #21): half of int8's memory,\n"
                 "                       slightly lower precision (see bench/results/2026-09-27-kv-q4)\n"
                 "  --kv k8v4            hybrid: INT8 K (exact attention scores) + rotated Q4_0 V, 816 B/cell\n"
                 "                       (vs int8's 1,056); streams with --kv-resident too\n"
                 "  --kv-resident N      KV streaming: keep N cells of each QSA layer in VRAM (min 20480) and the\n"
                 "                       whole K/V in pinned RAM; the freed VRAM goes to expert slots. 0 (default):\n"
                 "                       all of it in VRAM. A context of N cells or fewer is not streamed\n"
                 "  --stream-token       enqueue token work on the session stream (experimental)\n"
                 "  --check-logits       copy and check all logits in the stream-token path\n"
                 "  --gr-fp32-activations  experimental CUDA-oracle GR activation precision\n"
                 "  --gr-native-mmvf      experimental pinned GR norm/projections; implies FP32 activations\n"
                 "  --native-bf16         experimental CUDA-oracle SSM/router/indexer BF16 projections\n"
                 "  --native-bf16-extra   experimental CUDA-oracle PLE/shared gate BF16 projections\n"
                 "  --native-ple-key      experimental native PLE key; requires --native-dense-gguf\n"
                 "  --native-moe-combine  experimental pinned CUDA routed/shared combination\n"
                 "  --native-gdn          experimental pinned CUDA GDN norms/gates/recurrence\n"
                 "  --native-flash-attn-short  diagnostic pinned vector attention; --max-context <=256\n"
                 "  --native-qsa-indexer  experimental pinned indexer key cache and pooling\n"
                 "  --native-qsa          experimental pinned QSA normalization and output gate\n"
                 "  --native-rope         experimental pinned text-only CUDA rotary arithmetic\n"
                 "  --native-ple-postops  experimental pinned PLE postprojection arithmetic\n"
                 "  --native-router       experimental pinned CUDA 512-expert top-10 routing\n"
                 "  --cpu-oracle-q8-0     experimental pinned CPU expert quantization and dot reduction\n"
                 "  --native SHARD1      every full-context native path at once (plan v0.3 P1): stream-token,\n"
                 "                       GR MMVF, BF16, head, dense + PLE key, MoE combine, GDN, router, QSA,\n"
                 "                       indexer, RoPE, PLE postops, and the CPU q8_0 contract unless the\n"
                 "                       expert cache is on. Individual --native-* flags stay for A/B.\n"
                 "  --native-head-gguf PATH  native Q5_K head from model shard 1; requires --stream-token\n"
                 "  --native-dense-gguf PATH native GDN/QSA/shared projections; repeat for each source model shard\n"
                 "  --expert-cache-cpu-order  experimental GPU expert reduction matching CPU order\n"
                 "  --max-new N          tokens to generate (default 16)\n"
                 "  --max-context N      KV/state capacity (default 4096)\n"
                 "  --rope-scaling T     extend the context past the trained one: none, linear (position\n"
                 "                       interpolation) or yarn - llama.cpp's types.  Default: the model file's\n"
                 "                       rope keys, else none.  Fixed at startup (K in the cache is post-RoPE)\n"
                 "  --rope-scale F       the extension factor for linear/yarn (default: the model file's, else 1)\n"
                 "  --rope-freq-base N   the frequency base (0 = the model's 1e7) and the angle shrink\n"
                 "  --rope-freq-scale F  (0 = 1/--rope-scale)\n"
                 "  --yarn-orig-ctx N    the trained context the correction targets (0 = 262144)\n"
                 "  --yarn-ext-factor F --yarn-attn-factor F --yarn-beta-fast F --yarn-beta-slow F\n"
                 "                       YaRN's knobs; defaults -1 (auto: 1 for yarn), 1, 32, 1\n"
                 "  --greedy             argmax (the default)\n"
                 "  --seed S             enable sampling with this Philox seed\n"
                 "  --top-k N --top-p F --temperature F\n"
                 "  --dump-logits PATH   write one line of raw logits per position (a native pack: from the\n"
                 "                       last prompt token on; its prompt is read batched)\n"
                 "  --logits-stride N    store every Nth row plus final input (default 1); N>1 requires --max-new 1\n"
                 "  --dump-residual PATH write the final R (hc x n_embd, f32) for head bisection\n"
                 "  --dump-layers PATH   write R after EVERY layer, per position: the C1 bisection ladder\n"
                 "  --dump-halves PATH   write both halves' block_out and inject per layer: the half bisection\n"
                 "  --dump-routing PATH  write the routed expert ids and weights per layer per position (P0.S8)\n"
                 "  --no-capture         run the layers directly instead of replaying graphs\n"
                 "  --shared-late        A/B: shared expert after the CPU pool (default: overlapped with it)\n"
                 "  --keep-canonical     A/B: also load canonical copies of natively served tensors (more VRAM)\n"
                 "  --vision             --serve takes images too (GENI requests; embeddings from strata-vision)\n"
                 "  --prompt-cache N     --serve: keep N conversation checkpoints between requests (default 6, ~118 MB\n"
                 "                       of RAM each; 0 = read every prompt from the start)\n"
                 "  --conversation-cache-mib N  --serve: RAM budget for parked conversations (default 0 = off)\n"
                 "  --conversation-cache-slots N  --serve: at most N parked conversations (default 4)\n"
                 "  --conversation-cache-min-free-mib N  --serve: physical RAM floor when parking or restoring a\n"
                 "                       session file (default 2560)\n"
                 "  --session-min-free-mib N  --serve: disk space a SAVE must leave free (default 4096; 0 = no check)\n"
                 "  --conversation-save DIR  --serve: also keep parked conversations in DIR across restarts (default off;\n"
                 "                       needs --conversation-cache-mib)\n"
                 "  --conversation-save-checkpoints N  --serve: checkpoints written per conversation (default 2)\n"
                 "  --conversation-save-mib N  --serve: at most N MiB of files in DIR, oldest deleted first (default 16384)\n"
                 "  --conversation-save-hours N  --serve: delete a file N hours after its last use (default 48)\n"
                 "  --conversation-save-compress  --serve: compress the files' floating-point parts (c-blosc2 ZSTD;\n"
                 "                       10-15%% smaller; only in a build that found libblosc2)\n"
                 "  --prompt-cache-every N  --serve: also checkpoint every N fresh prompt tokens (default 16384, 0 = off)\n"
                 "  --prompt-cache-tail  --serve, single GPU: one extra checkpoint near the prompt's end, at an existing\n"
                 "                       chunk boundary (default off; requires --prompt-cache and --prompt-cache-every > 0)\n"
                 "  --turn-token ID      --serve: the token that opens a chat turn (default 248045, <|im_start|>)\n"
                 "  --tail-role-token ID --serve: a turn of this role right before the last turn is left out of\n"
                 "                       the conversation checkpoint (a trailing effort turn; default -1 = off)\n"
                 "  --short-read N       --serve: read at most N fresh text tokens through the decode windows instead\n"
                 "                       of the batched prompt path (default 64, 0 = off)\n"
                 "  --suffix-draft N     prompt lookup: draft from an earlier repeat of the last N+ tokens of context\n"
                 "                       when it pays (default 3; 0 = MTP only)\n"
                 "  --mtp-max-t M        cap the MTP's windows at M tokens (0 = --spec; longer ones come from suffixes)\n"
                 "  --spec N             the MTP drafter's verify window: how many tokens it proposes per check (setup\n"
                 "                       writes 4; --suffix-draft lets it grow by 2, up to 8, where a repeat is likely)\n"
                 "  --spec-min-p P       how sure the draft layer must be to extend a verify window by another guess\n"
                 "                       (setup writes 0.5; --calibrate measures it on this PC, see docs/DETAILS.md)\n"
                 "  --lookup-chain K     opt-in: after the MTP's drafts, add up to K prompt-lookup drafts that continue\n"
                 "                       them (the window grows to at most 8; default 0 = off)\n"
                 "  --lookup-chain-min M  the shortest context match --lookup-chain extends on (default 3)\n"
                 "  --mtp-hnorm pooled|stream  the draft layer's hidden-input norm: one RMS over all four streams\n"
                 "                       (default) or one per stream (llama.cpp's MTP graph)\n"
                 "  --control-vector-scaled FILE:SCALE[,...]  a control vector GGUF on the residual stream (llama.cpp's\n"
                 "                       format; --control-vector FILE = scale 1).  --serve: requests switch it (cvec=0|1)\n"
                 "  --control-vector-layer-range A B  the layers it follows (inclusive; default 1 .. the last)\n"
                 "  --cvec-mode add|project  h += s v (default) or h -= s (h.v) v with v unit\n"
                 "  --cvec-dir per-layer|single:L  each layer's own direction (default) or layer L's everywhere (project)\n"
                 "  --no-token-graph     A/B: two graphs per layer (the host launches each) instead of one per token\n"
                 "  --no-fused-gr        A/B: the six-kernel hyper-connection read and a separate write (native)\n"
                 "  --prefill CHUNK      batched prompt processing in chunks of CHUNK tokens (needs --native); auto =\n"
                 "                       the largest chunk up to 32768 whose buffers the expert cache can lend\n"
                 "  --prefill-experts G  experts the prompt path dequantizes before one grouped product (1-64,\n"
                 "                       default 16; 9.4 MiB of VRAM each, taken from the lent cache slots)\n"
                 "  --no-pool            skip the CPU expert pool (the GPU-only floor)\n"
                 "  --sync-every-layer   debug: synchronise after every layer\n"
                 "  --ple-gguf PATH      the n-gram/PLE shard.  WITHOUT IT LAYER 1's PLE IS SILENTLY SKIPPED,\n"
                 "                       which changes every number downstream - pass it for any real run\n"
                 "  --dump-mixed PATH    write the post-attention residual (n_embd, f32)\n"
                 "  --stage-timing       per-stage KERNEL-COUNT shares.  NOT a time profile: an uncaptured\n"
                 "                       event interval includes host gaps, so run with --gpu-only-full first\n"
                 "  --graph-only         MEASURE: replay the 48 `pre` graphs only.  OMITS the 48 `post` graphs\n"
                 "                       and the LM head, so it is NOT the GPU floor (R0.3, Memory/ERRORS.md A4)\n"
                 "  --gpu-only-full      MEASURE: replay pre+post for all 48 layers plus the LM head, no pool.\n"
                 "                       THE TRUE PER-TOKEN GPU FLOOR.  Quote this one, not --graph-only.\n"
                 "  --stats              print the per-stage breakdown\n"
                 "  --spec-follow PATH   benchmarks: emit this continuation (token ids) instead of the argmax and\n"
                 "                       accept the drafts that match it, so speculation settings compare on the\n"
                 "                       same text (with --spec; not --serve)\n"
                 "  --window-hashes PATH per verify window, a line with its index, position, size, a 64-bit hash of\n"
                 "                       its final residual rows and its argmaxes: two builds that do the same\n"
                 "                       arithmetic write the same file (use with --spec-follow, --adapt-every 0)\n"
                 "  --gpu-stages         R0.9: capture the layer as three graphs (mixer / ffn+router / post)\n"
                 "                       and time them from OUTSIDE the capture.  The per-stage table on the\n"
                 "                       real graph that --stage-timing cannot give.  Prints and exits.\n"
                 "  --expert-profile P   R4.2e: pre-load the VRAM tier from a `profile.bin` (see\n"
                 "                       tools/make_profile.py) instead of admitting on first use.\n"
                 "  --expert-profile-save P  --serve, #477: save what the adaptive tier learned (the experts in\n"
                 "                       VRAM, then the routing counted since the start) as a profile at P, on\n"
                 "                       QUIT and every --expert-profile-save-every MIN minutes (default 10;\n"
                 "                       0 = on QUIT only) between requests; start from it with --expert-profile P\n"
                 "  --no-hit-poke        R4.2d's A/B arm.  The hit path pokes the driver once right after its\n"
                 "                       launch so the GPU starts while the CPU pool runs; without it the work\n"
                 "                       waits for the next driver entry and does not overlap at all.\n"
                 "  --expert-cache N     R4: keep N expert blobs resident in VRAM and compute their rows on the\n"
                 "                       GPU via `moe_hit_grouped_s2`.  DEFAULT 0.  Measured at 4096 slots\n"
                 "                       with --expert-cache-per-layer: 54.4%% hits, CPU pool drain 19.1 -> 10.3\n"
                 "                       ms/token, -2.7 ms/token end to end.\n"
                 "  --expert-cache-device1 N  pre-fill N experts on CUDA1 (experimental)\n"
                 "  --expert-cache-device2 N  pre-fill N more experts on CUDA2\n"
                 "  --expert-cache-device3 N  pre-fill N more experts on CUDA3\n"
                 "  --expert-cache-remote-placement stripe|layer  distribute expert ranks or whole\n"
                 "                       layers across CUDA1..3 (default: stripe)\n"
                 "  --batch N, --slots N --serve: up to N conversations decoded together, one token each per batch\n"
                 "                       window (2..8; a count that cannot run is a warning and fewer slots or none)\n"
                 "  --layer-split K[,K2..]|auto  --serve: the next GPU runs layers from K on (docs/MULTIGPU.md);\n"
                 "                       auto chooses K from what this PC measures, before loading\n"
                 "  --gpu ADDR           the GPU by PCI address (domain:bus:device.function), as STRATA_GPU_PCI\n"
                 "  --split-device D[,D2..]  the GPU for each split point, by engine number or PCI address\n"
                 "                       (default: the other discrete GPUs)\n"
                 "  --batch-groups G     with a layer split: the --batch slots in G groups pipelined through the GPUs\n"
                 "                       (each GPU on another group at once; G divides N; 1 = one window, GPU\n"
                 "                       after GPU)\n"
                 "  --batch-cpu-split N0,N1  EXPERIMENTAL, with --batch-groups over two GPUs: the CPU expert workers\n"
                 "                       in two pools (N0 + N1) so both stages' CPU experts run at once (slower where\n"
                 "                       the RAM bandwidth is the limit, as on a DDR PC; for PCs with more of it)\n"
                 "  --vram-elastic       --serve (#533, opt-in): the expert cache in segments (--vram-segment-mib,\n"
                 "                       default 512), so the command `VRAM <reserve_mib>` (the server's POST /v1/vram) can\n"
                 "                       give VRAM back to other programs between requests and take it back later\n"
                 "  --expert-cache-per-layer  R4.2g: give each layer its OWN slots instead of letting the first\n"
                 "                       position take all of them.  The default policy fills in arrival order\n"
                 "                       from one shared counter, so 256 slots went to ~26 layers of position 0\n"
                 "                       and measured **2.97%%**.  Per-layer, the same routing gives 21.4%% at 8\n"
                 "                       slots/layer and 70.4%% at 64.\n"
                 "  --no-host-worker     R2.2: the A/B arm.  By default the HOST THREAD joins the drain, so the\n"
                 "                       pool is six threads on six cores instead of five plus an idle core;\n"
                 "                       this flag restores the five-worker form for comparison on `pool phases`.\n"
                 "  --pool-workers N     R2.2: CPU expert pool worker count.  Default 0 = every physical core\n"
                 "                       except the one the host loop spins on.  A sweep is how the pool's\n"
                 "                       deviation from `cpu_s2` is attributed.\n"
                 "  --pool-tasks N       Batched CPU expert tasks per GU/Down phase (0..4096). Default 0 =\n"
                 "                       3 per participating thread; positive counts are capped by row count.\n"
                 "  --mmap-experts       the low-RAM mode: the experts from the files through the OS page cache\n"
                 "                       (experts.bin, or a native pack's GGUF files in place) instead of a\n"
                 "                       copy of them all in RAM; the GPU cache holds the most used\n"
                 "  --resident-budget-gib N  with it: the experts the GPU cache does not hold copied into RAM\n"
                 "                       in the profile's order while they fit N GiB (at most the free RAM\n"
                 "                       less 8 GiB); the rest are read from the files\n"
                 "  --resident-experts   the same with as much RAM as is free (less 8 GiB)\n"
                 "  --resident-cpu-experts  cache the static GPU cache's misses in ordinary RAM (with --mmap-experts)\n"
                 "  --shared-expert-arena FILE  share completed expert weights between Linux processes\n"
                 "  --coupled-draft / --no-coupled-draft  sample MTP drafts with the target's chain (default: off)\n");
}

/// A blob for a copy that is not over at once (a queued or asynchronous fill, a long loop of blocking ones): the
/// source's own pointer, or - for a source that assembles its blobs into short-lived buffers (the GGUF read in place,
/// `transient`) - a copy in `tmp`, which the caller must not reuse before the copy from it is done.
const uint8_t* blob_to_copy(strata::core::ExpertSource* src, int64_t layer, int64_t expert, std::vector<uint8_t>& tmp) {
    if (!src->transient(layer, expert)) return src->blob(layer, expert);
    tmp.resize((size_t) strata::kernels::cpu::expert_layout().max_blob);
    return src->copy_blob(layer, expert, tmp.data()) ? tmp.data() : nullptr;
}

// --lookup-chain: the context's last tokens for the draft sources (at most 64), the MTP's pending drafts last
int chain_tail(const strata::spec::SuffixDrafter& sfx, const int32_t* pending, int n_pending, std::vector<int32_t>& tail) {
    const std::vector<int32_t>& h = sfx.history();
    const int keep = std::max(0, std::min((int) h.size(), 64 - n_pending));
    tail.assign(h.end() - keep, h.end());
    tail.insert(tail.end(), pending, pending + n_pending);
    return (int) tail.size();
}

// --lookup-chain: the registered extra draft sources (strata/spec/draft_source.hpp) see the committed context too
void extra_sources_reset() {
    for (auto& src : strata::spec::extra_draft_sources()) src->reset();
}
void extra_sources_append(const int32_t* t, size_t n) {
    for (auto& src : strata::spec::extra_draft_sources()) src->append(t, n);
}

bool parse_i64_list(const char* s, std::vector<int64_t>& out, std::string& err) {
    out.clear();
    std::string text(s);
    for (char& c : text) if (c == ',') c = ' ';
    std::istringstream input(text);
    std::string token;
    while (input >> token) {
        int32_t id = 0;
        const auto parsed = std::from_chars(token.data(), token.data() + token.size(), id);
        if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() || id < 0) {
            err = "invalid token id: expected an integer in [0, 2147483647]";
            out.clear();
            return false;
        }
        out.push_back(id);
    }
    if (out.empty()) { err = "token list was empty"; return false; }
    return true;
}

/// The pool's adapter plus the wall-clock it spent, so the report can say how much of the token was the CPU.
struct Drive {
    strata::core::ExpertDispatch d;
    double cpu_ms = 0;
    int64_t calls = 0;
    /// THE ROUTING TRACE, which is P0.S8 and a stated prerequisite of Phase 3.  `drive_pool` is called once
    /// per layer from the main loop - the workers live inside `expert_pool_dispatch` - so a single FILE* here
    /// needs no locking.  `d.layers` is the CURRENT layer on entry (the adapter increments it as it walks the
    /// blob), which is why the layer index comes from there rather than from a counter of our own.
    std::FILE* routing = nullptr;
};

void drive_pool(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd, int64_t k,
                float* out) {
    Drive* t = (Drive*) user;
    const Clock::time_point a = Clock::now();
    strata::core::expert_pool_dispatch(&t->d, x_f, ids, weights, n_embd, k, out);
    t->cpu_ms += std::chrono::duration<double, std::milli>(Clock::now() - a).count();
    ++t->calls;
    // THE ROUTING TRACE.  Written AFTER the dispatch so the layer index is still this layer's: `d.layers` is
    // advanced by the adapter as it consumes the blob, and reading it after the call is the same value the
    // dispatch used.  Record = int32 layer, int32 k, k int32 ids, k float weights.
    if (t->routing != nullptr) {
        // **`d.layers` HAS ALREADY BEEN ADVANCED BY THE TIME THIS RUNS, AND THE FIRST TRACE WAS OFF BY ONE
        // BECAUSE OF IT.**  The adapter walks the blob by incrementing `d.layers` as it consumes each layer's
        // experts, so after the dispatch it holds the NEXT layer's index.  `tools/make_profile.py` caught it
        // with a bounds check when the trace turned out to span 1..48 instead of 0..47.  The hit-rate CURVE was
        // unaffected - it is a per-layer split, and shifting every layer by one preserves both metrics - but
        // anything keyed on the layer index, which is exactly what a cache profile is, would have been wrong.
        const int32_t layer_idx = (int32_t) (t->d.layers - 1);
        if (layer_idx < 0 || layer_idx >= 48) {
            std::fprintf(stderr, "strata generate: the routing trace saw layer %d, outside 0..47\n", layer_idx);
            return;
        }
        const int32_t rec[2] = {layer_idx, (int32_t) k};
        std::fwrite(rec, sizeof rec, 1, t->routing);
        std::fwrite(ids, sizeof(int32_t), (size_t) k, t->routing);
        std::fwrite(weights, sizeof(float), (size_t) k, t->routing);
    }
}

/// Plan v0.3 P6: the pool for a verify window.
void drive_pool_multi(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                      int64_t layer) {
    Drive* t = (Drive*) user;
    t->d.layers = layer;
    const Clock::time_point a = Clock::now();
    strata::core::expert_pool_dispatch_multi(t->d, x_f, ids, n_tok, k, out);
    t->cpu_ms += std::chrono::duration<double, std::milli>(Clock::now() - a).count();
    ++t->calls;
    // the routing trace for the serve path: the same record format drive_pool writes (layer, k, ids, weights),
    // one record per token.  The multi dispatch fuses the router weights into the kernel and does not surface
    // them, so records carry unit weights: tools/make_profile.py ranks pairs by routed frequency, which is the
    // signal that matters; a one-shot --dump-routing run records true weights if a weighted ranking is wanted.
    if (t->routing != nullptr && layer >= 0 && layer < 48) {
        for (int64_t tok = 0; tok < n_tok; ++tok) {
            const int32_t rec[2] = {(int32_t) layer, (int32_t) k};
            std::fwrite(rec, sizeof rec, 1, t->routing);
            std::fwrite(ids + tok * k, sizeof(int32_t), (size_t) k, t->routing);
            static const float one[64] = {};   // k <= 64 in a verify window; zeros read as unit weights
            std::fwrite(one, sizeof(float), (size_t) k, t->routing);
        }
    }
}

/// Layer split: every verify stage shares one Drive (its counters, usage and failure flags); the GPU plan, the expert
/// cache and the PCIe share the pool uses for a layer are those of the stage that runs it.
struct SplitDrive {
    static constexpr int kMax = 8;
    Drive* base = nullptr;
    int n = 0;                                    ///< stages
    int64_t end[kMax] = {};                       ///< stage i runs the layers from end[i - 1] (0) below end[i]
    strata::core::GpuPlanSink* plan[kMax] = {};
    const uint8_t* cache_base[kMax] = {};
    const uint64_t* cache_slot_off[kMax] = {};
    int pcie_num[kMax] = {};
};
void drive_pool_split(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                      int64_t layer) {
    SplitDrive* s = (SplitDrive*) user;
    int st = 0;
    while (st + 1 < s->n && layer >= s->end[st]) ++st;
    Drive& d = *s->base;
    d.d.plan = s->plan[st];
    d.d.cache_base = s->cache_base[st];
    d.d.cache_slot_off = s->cache_slot_off[st];
    d.d.pcie_num = s->pcie_num[st];
    drive_pool_multi(s->base, x_f, ids, n_tok, k, out, layer);
}

/// Layer split across GPUs: a later stage on its own device, with its own copy of the dense weights, a session, an
/// expert cache for its layers, a verify window and a prompt path; the last one also holds the head (the drafter
/// lives on its device too).
struct GpuStage {
    int dev = 0;
    int64_t lb = 0, le = 0;
    double pcie_frac = 0.0;
    strata::core::WeightTable wt;
    strata::core::NativeDense dense;
    strata::core::NativeHead head;
    strata::core::SessionState ss;
    strata::gpu::Stream stream = nullptr;
    strata::core::ExpertCache cache;
    std::vector<std::pair<int32_t, int32_t>> profile;   ///< its layers' share of the profile, hottest first
    int64_t held = 0;                                    ///< the first `held` pairs of `profile` fill its cache
    std::vector<std::unique_ptr<strata::core::SessionState>> bslots;   ///< --batch: the slots' sessions on its GPU
    std::vector<void*> bslot_buf;
    int32_t* d_res = nullptr;                            ///< the residency table on its device
    strata::core::Verifier ver;
    strata::prefill::Prefill sp;
    strata::gpu::Stream adapt_stream = nullptr;
    strata::gpu::Event* adapt_ev = nullptr;
    bool adapt_live = false;                             ///< swaps of this request are in flight on it
    int32_t* mrope = nullptr;                            ///< --vision: the image-position table on its device
};

// ---- issue #31: what the watchdog prints before it stops a stalled engine
struct MemSample {
    unsigned long long faults = 0, rss_mib = 0, avail_mib = 0, commit_mib = 0;
};
MemSample mem_sample() {
    MemSample m;
    if (std::FILE* f = std::fopen("/proc/self/stat", "r")) {
        char buf[4096];
        const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
        buf[n] = 0;
        std::fclose(f);
        const char* s = std::strrchr(buf, ')');   // fields after the command name: 3 state ... 12 majflt
        for (int field = 2; s && field < 12; ++field) s = std::strchr(s + 1, ' ');
        if (s) m.faults = std::strtoull(s + 1, nullptr, 10);
    }
    auto kb = [](const char* path, const char* key) -> unsigned long long {
        unsigned long long v = 0;
        if (std::FILE* f = std::fopen(path, "r")) {
            char line[256];
            const size_t kl = std::strlen(key);
            while (std::fgets(line, sizeof line, f))
                if (std::strncmp(line, key, kl) == 0) { v = std::strtoull(line + kl, nullptr, 10); break; }
            std::fclose(f);
        }
        return v;
    };
    m.rss_mib = kb("/proc/self/status", "VmRSS:") >> 10;
    m.commit_mib = kb("/proc/self/status", "VmSwap:") >> 10;
    m.avail_mib = kb("/proc/meminfo", "MemAvailable:") >> 10;
    return m;
}

// the stage the watchdog names: "<where> <detail>", and the prompt chunk a batched read is in (upstream 9631c96)
std::string stage_text() {
    const strata::core::Progress& p = strata::core::progress();
    std::string s = std::string(p.where.load()) + " " + std::to_string((long long) p.detail.load());
    if (const int64_t c = p.chunk.load(); c >= 0) s += " of the prompt chunk from token " + std::to_string((long long) c);
    return s;
}

void stall_report(std::FILE* f, uint64_t layers_during) {
    strata::core::Progress& p = strata::core::progress();
    std::fprintf(f, "strata serve: stall report (engine %s): stage \"%s\" for %lld s; %llu layers served since the "
                    "last finished step (0 = stopped, more = slow)\n", STRATA_VERSION, stage_text().c_str(),
                 (long long) ((strata::core::progress_now_ms() - p.since_ms.load()) / 1000),
                 (unsigned long long) layers_during);
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            std::fprintf(f, "  2 s later:\n");
        }
        if (auto fn = strata::core::diag_pool_fn().load()) fn(f);
        if (auto fn = strata::core::diag_verify_fn().load()) fn(f);
        const MemSample m = mem_sample();
        std::fprintf(f, "  memory: %llu MiB resident, %llu MiB %s, %llu MiB RAM available; %llu %s\n", m.rss_mib,
                     m.commit_mib,
                     "in swap", m.avail_mib, m.faults, "major page faults so far"
        );
        std::fflush(f);
    }
}

/// #463's A/B: STRATA_ADAPT_NOWAIT=1 lets a verify window start before the adaptive tier's copies have landed (0.1.37)
bool adapt_nowait() {
    static const bool v = [] { const char* e = std::getenv("STRATA_ADAPT_NOWAIT"); return e && e[0] == '1'; }();
    return v;
}

/// STRATA_TRACE=1: the VRAM left at a step of the startup (finds what fills the card after the cache is sized)
void mem_mark(const char* where) {
    static const bool on = std::getenv("STRATA_TRACE") != nullptr;
    if (!on) return;
    size_t free_b = 0, total_b = 0;
    strata::gpu::mem_info(&free_b, &total_b);
    std::fprintf(stderr, "strata trace: %lld MiB free after %s\n", (long long) (free_b >> 20), where);
}

// ---- --serve's conversation cache.  A chat or an agent sends the whole conversation again with every request, and
// reading it again is what made a long session wait minutes for every turn.  What a sequence leaves behind splits in
// two, and only one half needs copying:
//   * POSITIONAL state - the KV cache of the 12 QSA layers and their pooled indexer keys, the draft layer's KV.  A
//     cell is written once for its position and read only by later positions (the block scores take `dead` for the
//     block being filled, never its pooled row), so rewinding to a position just means writing from there again.
//   * RUNNING state - the 36 GDN recurrences and conv histories, each QSA layer's indexer tail (the unfinished
//     block's raw keys) and the PLE's normalized history.  Each describes "everything so far" and cannot be
//     rewound, so a checkpoint is a copy of exactly these: ~118 MB, the same set the verifier snapshots to roll
//     back rejected drafts.
// A checkpoint is only valid while the positional cells below it still hold ITS tokens, so the serve loop keeps
// just the checkpoints that are prefixes of the tokens the session holds now.
using ImgKey = strata::core::ConversationImageKey;
using ConvCheckpoint = strata::core::ConversationCheckpoint;

uint64_t fnv1a(const void* data, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* p = (const uint8_t*) data;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

using ConvStateSizes = strata::core::ConversationStateSizes;

ConvStateSizes conv_state_sizes(const strata::core::ModelGeometry& g) {
    ConvStateSizes z;
    std::string error;
    strata::core::conversation_state_sizes(g, z, error);   // the geometry has passed the engine's checks already
    return z;
}

/// Copies the running state out.  The caller has synchronized the device.
bool checkpoint_save(ConvCheckpoint& c, const strata::core::SessionState& ss, const strata::core::ModelGeometry& g) {
    std::string error;
    if (strata::core::conversation_checkpoint_save(c, ss, g, error)) return true;
    std::fprintf(stderr, "strata serve: checkpoint save: %s\n", error.c_str());   // the caller's ERR has no reason
    return false;
}

/// Puts a checkpoint's running state back; the positional cells below it are the caller's to guarantee.
bool checkpoint_restore(const ConvCheckpoint& c, strata::core::SessionState& ss, const strata::core::ModelGeometry& g) {
    std::string error;
    if (strata::core::conversation_checkpoint_restore(c, ss, g, error)) return true;
    std::fprintf(stderr, "strata serve: checkpoint restore: %s\n", error.c_str());
    return false;
}

// --control-vector-scaled: llama.cpp's `common_control_vector_load` (every file's `direction.<l>` times its scale,
// summed; layer 0 has none) and `llama_adapter_cvec::apply` with the projection-mode patch (project: the unit
// direction and its norm as the scale), into the tables `cvec_upload` takes.  `summary` is what INFO reports;
// `digest` identifies the uploaded tables, mode and range exactly (a session file is bound to it).
bool load_control_vectors(const Options& o, const strata::core::ModelGeometry& g, std::string& summary,
                          uint64_t& digest, std::string& err) {
    const int64_t L = g.n_layers, N = g.n_embd;
    std::vector<float> data((size_t) (L * N), 0.0f);
    std::vector<bool> have((size_t) L, false);
    for (const auto& [path, scale] : o.cvec_files) {
        try {
            strata::GgufFile f(path);
            const strata::MetaValue* arch = f.get("general.architecture");
            if (arch == nullptr || arch->s != "controlvector") {
                err = path + ": not a control vector GGUF (general.architecture is not 'controlvector')";
                return false;
            }
            const strata::MetaValue* hint = f.get("controlvector.model_hint");
            if (hint != nullptr && hint->s != "qwen4exp")
                std::fprintf(stderr, "strata generate: %s was made for '%s', not qwen4exp\n", path.c_str(), hint->s.c_str());
            int found = 0;
            for (const strata::TensorInfo& t : f.tensors()) {
                if (t.name.rfind("direction.", 0) != 0) continue;
                const long l = std::strtol(t.name.c_str() + 10, nullptr, 10);
                if (l < 1 || l >= L) continue;   // layer 0 has no vector; past the model is ignored, as in llama.cpp
                if (t.type != 0 || t.elements() != (uint64_t) N) {
                    err = path + ": " + t.name + " must be " + std::to_string((long long) N) + " f32";
                    return false;
                }
                const float* src = reinterpret_cast<const float*>(f.tensor_data(t));
                for (int64_t j = 0; j < N; ++j) data[(size_t) (l * N + j)] += scale * src[j];
                have[(size_t) l] = true;
                ++found;
            }
            if (found == 0) { err = path + ": no direction.<layer> tensors"; return false; }
        } catch (const std::exception& e) {
            err = e.what();
            return false;
        }
    }
    const int first = o.cvec_first <= 0 ? 1 : o.cvec_first;
    const int last = (o.cvec_last <= 0 || o.cvec_last >= L) ? (int) L - 1 : o.cvec_last;
    const int single = o.cvec_mode == 0 ? o.cvec_single : -1;
    if (single >= 0 && (single >= L || !have[(size_t) single])) {
        err = "--cvec-dir single:" + std::to_string(single) + ": the vector has no direction for that layer";
        return false;
    }
    std::vector<float> dir((size_t) (L * N), 0.0f), s((size_t) L, 0.0f);
    int steered = 0;
    for (int64_t l = first; l <= last; ++l) {
        const int64_t src = single >= 0 ? single : l;
        if (!have[(size_t) src]) continue;
        const float* d = data.data() + (size_t) (src * N);
        if (o.cvec_mode == 0) {
            double nrm = 0.0;
            for (int64_t j = 0; j < N; ++j) nrm += (double) d[j] * d[j];
            nrm = std::sqrt(nrm);
            if (nrm <= 0.0) continue;
            s[(size_t) l] = (float) nrm;
            for (int64_t j = 0; j < N; ++j) dir[(size_t) (l * N + j)] = (float) (d[j] / nrm);
        } else {
            s[(size_t) l] = 1.0f;
            std::copy(d, d + N, dir.begin() + (size_t) (l * N));
        }
        ++steered;
    }
    if (steered == 0) { err = "the control vector has no direction in layers " + std::to_string(first) + ".." + std::to_string(last); return false; }
    if (!strata::kernels::cvec_upload(dir, s, o.cvec_mode, first, last, N, g.hc, err)) return false;
    {
        strata::core::SessionIdentityBuilder b(0x43564543ull);   // "CVEC"
        b.i64("mode", o.cvec_mode);
        b.i64("first", first);
        b.i64("last", last);
        b.i64("single", single);
        b.bytes("dir", dir.data(), dir.size() * sizeof(float));
        b.bytes("scale", s.data(), s.size() * sizeof(float));
        digest = b.digest();
        if (digest == 0) digest = 1;   // 0 means "none loaded"
    }
    summary = std::string(o.cvec_mode == 0 ? "project" : "add") + ":" + std::to_string(first) + "-" + std::to_string(last) +
              (single >= 0 ? ":single" + std::to_string(single) : "");
    // the line llama.cpp's patched build prints, so a log shows the same thing
    std::fprintf(stderr, "strata generate: control vector mode = %s, dir = %s, layers %d..%d (%d steered)\n",
                 o.cvec_mode == 0 ? "project" : "add", single >= 0 ? "single" : "per-layer", first, last, steered);
    return true;
}

// The effective host->device bandwidth of the PCIe link: copies from pinned host memory, as the expert arena's
// reads are.  The native default share (0.55) was measured on x16 links (~26-28 GB/s); a x8 card in a x8 slot
// carries about half of that.  Returns < 0 when the probe cannot run (then the caller keeps the default).
// #485 (upstream 4e308eb): a single timed burst can read a link low (a link still in a low-power state, other DMA, a
// context not yet up to speed), and the low reading set the share.  Such a disturbance only ever slows a copy, so the
// same 1 GiB is copied as four bursts of 256 MiB, timed one by one, and the fastest is the link's figure.  `samples`,
// when given, gets every burst's reading for the log.
double probe_pcie_h2d_gbps(std::string* samples = nullptr) {
    constexpr size_t kBytes = 256ull << 20;
    constexpr int kBursts = 4;
    void* h = nullptr;
    void* d = nullptr;
    if (!strata::gpu::alloc_host(&h, kBytes)) return -1.0;
    if (!strata::gpu::alloc_device(&d, kBytes)) {
        strata::gpu::free(h);
        return -1.0;
    }
    std::memset(h, 0, kBytes);   // fault the pages in before timing
    strata::gpu::copy_async(d, h, kBytes, nullptr);   // warmup: context up, copy engine primed
    // the bursts run back to back on the stream, an event between each two
    strata::gpu::Event* ev[kBursts + 1] = {};
    int n_ev = 0;
    while (n_ev <= kBursts && strata::gpu::event_create(&ev[n_ev])) ++n_ev;
    bool ok = n_ev == kBursts + 1;
    float ms[kBursts] = {};
    if (ok) {
        strata::gpu::event_record(ev[0], nullptr);
        for (int b = 0; b < kBursts; ++b) {
            strata::gpu::copy_async(d, h, kBytes, nullptr);
            strata::gpu::event_record(ev[b + 1], nullptr);
        }
        ok = strata::gpu::event_sync(ev[kBursts]);
        for (int b = 0; b < kBursts && ok; ++b) ok = strata::gpu::event_elapsed_ms(&ms[b], ev[b], ev[b + 1]);
    }
    for (int i = 0; i < n_ev; ++i) strata::gpu::event_destroy(ev[i]);
    double bw = -1.0;
    if (samples != nullptr) samples->clear();
    for (int b = 0; b < kBursts && ok; ++b) {
        const double sv = ms[b] > 0.01f ? (double) kBytes / (ms[b] * 1e-3) / 1e9 : -1.0;
        bw = std::max(bw, sv);
        if (samples != nullptr) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%s%.1f", b == 0 ? "" : " ", sv);
            *samples += buf;
        }
    }
    strata::gpu::free(d);
    strata::gpu::free(h);
    return bw;
}

// The PCIe share of the missed experts for a link measured at `gbps`: `base` (the share measured on x16 links) from
// 20 GB/s up, and below that in proportion to the bandwidth, so the time the link spends on its share stays about
// what the x16 share costs.  Continuous (#485): before, 19.9 GB/s gave 0.42 and 20.0 the full 0.55 (and 4.0 GB/s
// gave 0.08, 3.9 none).
double pcie_frac_for_gbps(double gbps, double base) {
    return gbps <= 0.0 ? base : base * std::min(1.0, gbps / 20.0);
}

}  // namespace

int main(int argc, char** argv) {
    // **UNBUFFERED, BECAUSE THE INTERESTING OUTPUT IS THE OUTPUT BEFORE A CRASH.**  `stdout` redirected to a
    // pipe or a file is block-buffered, so a program that dies loses every line it had already printed - which
    // turns "it crashed at step 7" into "it crashed somewhere", and the difference is a debugging session.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    // Load every CUDA kernel when the context is created, before the expert cache takes the free VRAM.  With the
    // default lazy loading, a kernel first used mid-prompt (MMQ for IQ3_XXS at 64K+ on a 12 GB card) found no VRAM
    // left for its code and the engine ended ("out of memory: cudaFuncSetAttribute").  Costs ~30 MB of VRAM.
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) {
        setenv("CUDA_MODULE_LOADING", "EAGER", 0);
    }
    // --gpu ADDR: the GPU by PCI address from the command line instead of the caller's environment (upstream #852).
    // The runtime reads STRATA_GPU_PCI when it starts, so this has to happen before anything else; the option loop
    // below consumes the value again.
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == "--gpu") { setenv("STRATA_GPU_PCI", argv[i + 1], 1); break; }
    Options o;
    bool have_tokens = false;
    bool have_logits_stride = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", what); std::exit(2); }
            return argv[++i];
        };
        if (a == "--help" || a == "-h") { usage(); return 0; }
        else if (a == "--pack") o.pack = next("--pack");
        else if (a == "--tokens") {
            if (have_tokens) { std::fprintf(stderr, "supply one token input only\n"); return 2; }
            std::string e;
            if (!parse_i64_list(next("--tokens"), o.tokens, e)) { std::fprintf(stderr, "%s\n", e.c_str()); return 2; }
            have_tokens = true;
        }
        else if (a == "--tokens-file") {
            if (have_tokens) { std::fprintf(stderr, "supply one token input only\n"); return 2; }
            std::ifstream input(next("--tokens-file"));
            if (!input) { std::fprintf(stderr, "cannot open token file\n"); return 2; }
            std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
            if (input.bad()) { std::fprintf(stderr, "cannot read token file\n"); return 2; }
            std::string error;
            if (text.find('\0') != std::string::npos || !parse_i64_list(text.c_str(), o.tokens, error)) {
                std::fprintf(stderr, "malformed token file: %s\n", error.c_str()); return 2;
            }
            have_tokens = true;
        }
        else if (a == "--max-new") o.max_new = std::atoll(next("--max-new"));
        else if (a == "--max-context") o.max_context = std::atoll(next("--max-context"));
        else if (a == "--rope-scaling") o.rope_scaling = next("--rope-scaling");
        else if (a == "--rope-scale") o.rope_scale = std::strtod(next("--rope-scale"), nullptr);
        else if (a == "--rope-freq-base") o.rope_freq_base = std::strtod(next("--rope-freq-base"), nullptr);
        else if (a == "--rope-freq-scale") o.rope_freq_scale = std::strtod(next("--rope-freq-scale"), nullptr);
        else if (a == "--yarn-orig-ctx") o.yarn_orig_ctx = std::strtod(next("--yarn-orig-ctx"), nullptr);
        else if (a == "--yarn-ext-factor") o.yarn_ext_factor = std::strtod(next("--yarn-ext-factor"), nullptr);
        else if (a == "--yarn-attn-factor") o.yarn_attn_factor = std::strtod(next("--yarn-attn-factor"), nullptr);
        else if (a == "--yarn-beta-fast") o.yarn_beta_fast = std::strtod(next("--yarn-beta-fast"), nullptr);
        else if (a == "--yarn-beta-slow") o.yarn_beta_slow = std::strtod(next("--yarn-beta-slow"), nullptr);
        else if (a == "--greedy") o.greedy = true;
        else if (a == "--seed") { o.seed = (uint64_t) std::atoll(next("--seed")); o.greedy = false; }
        else if (a == "--top-k") o.top_k = std::atoi(next("--top-k"));
        else if (a == "--top-p") o.top_p = (float) std::atof(next("--top-p"));
        else if (a == "--temperature") o.temperature = (float) std::atof(next("--temperature"));
        else if (a == "--dump-logits") o.dump_logits = next("--dump-logits");
        else if (a == "--logits-stride") {
            if (have_logits_stride) { std::fprintf(stderr, "--logits-stride must be supplied only once\n"); return 2; }
            if (!strata::program::logits_selection::parse_stride(next("--logits-stride"), o.logits_stride)) {
                std::fprintf(stderr, "--logits-stride requires a positive decimal int64\n"); return 2;
            }
            have_logits_stride = true;
        }
        else if (a == "--dump-residual") o.dump_residual = next("--dump-residual");
        else if (a == "--dump-mixed") o.dump_mixed = next("--dump-mixed");
        else if (a == "--dump-layers") o.dump_layers = next("--dump-layers");
        else if (a == "--dump-halves") o.dump_halves = next("--dump-halves");
        else if (a == "--dump-routing") o.dump_routing = next("--dump-routing");
        else if (a == "--ple-gguf") o.ple_gguf = next("--ple-gguf");
        else if (a == "--no-ple") o.no_ple = true;
        else if (a == "--ple-io") o.ple_io = next("--ple-io");
        else if (a == "--embd-gguf") o.embd_gguf = next("--embd-gguf");
        else if (a == "--ple-row-cache") o.ple_row_cache = std::atoll(next("--ple-row-cache"));
        else if (a == "--ple-inflight") o.ple_inflight = std::atoi(next("--ple-inflight"));
        else if (a == "--ple-delay-us") o.ple_delay_us = std::atof(next("--ple-delay-us"));
        else if (a == "--ple-sync-submit") o.ple_sync_submit = true;
        else if (a == "--kv") o.kv = next("--kv");
        else if (a == "--kv-resident") o.kv_resident = std::atoll(next("--kv-resident"));
        else if (a == "--stream-token") o.stream_token = true;
        else if (a == "--check-logits") o.check_logits = true;
        else if (a == "--gr-fp32-activations") o.gr_fp32_activations = true;
        else if (a == "--gr-native-mmvf") o.gr_native_mmvf = true;
        else if (a == "--native-bf16") o.native_bf16 = true;
        else if (a == "--native-bf16-extra") o.native_bf16_extra = true;
        else if (a == "--native-ple-key") o.native_ple_key = true;
        else if (a == "--native-moe-combine") o.native_moe_combine = true;
        else if (a == "--native-gdn") o.native_gdn = true;
        else if (a == "--native-flash-attn-short") o.native_flash_attn_short = true;
        else if (a == "--native-qsa-indexer") o.native_qsa_indexer = true;
        else if (a == "--native-qsa") o.native_qsa = true;
        else if (a == "--native-rope") o.native_rope = true;
        else if (a == "--native-ple-postops") o.native_ple_postops = true;
        else if (a == "--native-router") o.native_router = true;
        else if (a == "--cpu-oracle-q8-0") o.cpu_oracle_q8_0 = true;
        else if (a == "--native") o.native_preset = next("--native");
        else if (a == "--native-head-gguf") o.native_head_gguf = next("--native-head-gguf");
        else if (a == "--native-dense-gguf") o.native_dense_gguf.push_back(next("--native-dense-gguf"));
        else if (a == "--no-capture") o.no_capture = true;
        else if (a == "--no-pool") o.no_pool = true;
        else if (a == "--sync-every-layer") o.sync_every_layer = true;
        else if (a == "--stage-timing") o.stage_timing = true;
        else if (a == "--graph-only") o.graph_only = true;
        else if (a == "--gpu-only-full") o.gpu_only_full = true;
        else if (a == "--pool-workers") o.pool_workers = std::atoi(next("--pool-workers"));
        else if (a == "--pool-tasks") {
            const char* v = next("--pool-tasks");
            char* end = nullptr;
            const long tasks = std::strtol(v, &end, 10);
            if (end == v || *end != '\0' || tasks < 0 || tasks > strata::kernels::cpu::ExpertPool::kMaxTasks) {
                std::fprintf(stderr, "strata generate: --pool-tasks expects an integer in 0..4096 (0 = automatic)\n");
                return 2;
            }
            o.pool_tasks = (int) tasks;
        }
        else if (a == "--no-host-worker") o.no_host_worker = true;
        else if (a == "--expert-cache") {
            const std::string v = next("--expert-cache");
            o.expert_cache = (v == "auto") ? -1 : std::atoi(v.c_str());
        }
        else if (a == "--expert-cache-device1") o.expert_cache_remote[0] = std::atoi(next("--expert-cache-device1"));
        else if (a == "--expert-cache-device2") o.expert_cache_remote[1] = std::atoi(next("--expert-cache-device2"));
        else if (a == "--expert-cache-device3") o.expert_cache_remote[2] = std::atoi(next("--expert-cache-device3"));
        else if (a == "--expert-cache-remote-placement")
            o.expert_cache_remote_placement = next("--expert-cache-remote-placement");
        else if (a == "--vram-elastic") o.vram_elastic = true;
        else if (a == "--vram-segment-mib") o.vram_segment_mib = std::strtoll(next("--vram-segment-mib"), nullptr, 10);
        else if (a == "--vram-reserve-mib") {
            o.vram_reserve_mib = (int) std::strtol(next("--vram-reserve-mib"), nullptr, 10);
            o.vram_reserve_given = true;
        }
        else if (a == "--vram-reserve-later-mib")
            o.vram_reserve_later_mib = (int) std::strtol(next("--vram-reserve-later-mib"), nullptr, 10);
        else if (a == "--prefill") {
            const std::string v = next("--prefill");
            o.prefill_auto = v == "auto";
            o.prefill_chunk = o.prefill_auto ? 8192 : std::atoll(v.c_str());
        }
        else if (a == "--no-split-rows") o.no_split_rows = true;
        else if (a == "--no-prefill-borrow") o.no_prefill_borrow = true;
        else if (a == "--prefill-until") o.prefill_until = std::atoll(next("--prefill-until"));
        else if (a == "--prefill-experts") o.prefill_experts = (int) std::strtol(next("--prefill-experts"), nullptr, 10);
        else if (a == "--dump-final-r") o.dump_final_r = next("--dump-final-r");
        else if (a == "--spec") o.spec = std::atoi(next("--spec"));
        else if (a == "--batch" || a == "--slots") o.batch = (int) std::strtol(next(a.c_str()), nullptr, 10);
        else if (a == "--batch-groups") o.batch_groups = (int) std::strtol(next(a.c_str()), nullptr, 10);
        else if (a == "--batch-cpu-split") {
            const char* v = next(a.c_str());
            char* e = nullptr;
            o.batch_cpu_split[0] = (int) std::strtol(v, &e, 10);
            o.batch_cpu_split[1] = e != v && *e == ',' ? (int) std::strtol(e + 1, nullptr, 10) : 0;
            if (o.batch_cpu_split[0] < 1 || o.batch_cpu_split[1] < 1) {
                std::fprintf(stderr, "strata generate: --batch-cpu-split takes N0,N1 (CPU workers for each stage)\n");
                return 2;
            }
        }
        else if (a == "--spec-oracle") o.spec_oracle = next("--spec-oracle");
        else if (a == "--spec-corrupt") o.spec_corrupt = std::atoi(next("--spec-corrupt"));
        else if (a == "--spec-follow") o.spec_follow = next("--spec-follow");
        else if (a == "--window-hashes") o.window_hashes = next("--window-hashes");
        else if (a == "--mtp") o.mtp = next("--mtp");
        else if (a == "--mtp-window") o.mtp_window = std::atoll(next("--mtp-window"));
        else if (a == "--pcie-frac") o.pcie_frac = std::atof(next("--pcie-frac"));
        else if (a == "--adapt-every") o.adapt_every = std::atoi(next("--adapt-every"));
        else if (a == "--adapt-decay") o.adapt_decay = std::strtof(next("--adapt-decay"), nullptr);
        else if (a == "--spec-min-p") o.spec_min_p = std::atof(next("--spec-min-p"));
        else if (a == "--stop-eos") o.stop_eos = true;
        else if (a == "--spec-split") o.spec_split = true;
        else if (a == "--layer-split") o.layer_split = next("--layer-split");
        else if (a == "--gpu") (void) next("--gpu");   // applied at startup, before the runtime
        else if (a == "--split-device") o.split_device = next("--split-device");
        else if (a == "--pcie-mode") o.pcie_mode = next("--pcie-mode");
        else if (a == "--serve") o.serve = true;
        else if (a == "--vision") o.vision = true;
        else if (a == "--prompt-cache") o.prompt_cache = std::max(0, std::atoi(next("--prompt-cache")));
        else if (a == "--conversation-cache-mib" || a == "--conversation-cache-slots" ||
                 a == "--conversation-cache-min-free-mib" || a == "--conversation-save-checkpoints" ||
                 a == "--conversation-save-mib" || a == "--conversation-save-hours" || a == "--session-min-free-mib") {
            const std::string value = next(a.c_str());
            int64_t parsed = 0;
            const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
            const int64_t limit = a == "--conversation-cache-slots" || a == "--conversation-save-checkpoints"
                                      ? INT32_MAX
                                  : a == "--conversation-save-hours" ? INT64_MAX / 3600
                                                                     : INT64_MAX / (int64_t{1024} * 1024);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || parsed < 0 || parsed > limit) {
                std::fprintf(stderr, "%s needs a nonnegative integer within range\n", a.c_str());
                return 2;
            }
            if (a == "--conversation-cache-mib") o.conversation_cache_mib = parsed;
            else if (a == "--conversation-cache-min-free-mib") o.conversation_cache_min_free_mib = parsed;
            else if (a == "--conversation-save-checkpoints") o.conversation_save_checkpoints = (int) parsed;
            else if (a == "--conversation-save-mib") o.conversation_save_mib = parsed;
            else if (a == "--conversation-save-hours") o.conversation_save_hours = parsed;
            else if (a == "--session-min-free-mib") o.session_min_free_mib = parsed;
            else o.conversation_cache_slots = (int) parsed;
        }
        else if (a == "--conversation-save") o.conversation_save = next("--conversation-save");
        else if (a == "--conversation-save-compress") o.conversation_save_compress = true;
        else if (a == "--prompt-cache-tail") o.prompt_cache_tail = true;
        else if (a == "--prompt-cache-every") o.prompt_cache_every = std::max(0LL, std::atoll(next("--prompt-cache-every")));
        else if (a == "--prompt-cache-root") o.prompt_cache_root = std::max(0LL, std::atoll(next("--prompt-cache-root")));
        else if (a == "--turn-token") o.turn_token = std::atoll(next("--turn-token"));
        else if (a == "--tail-role-token") o.tail_role_token = std::atoll(next("--tail-role-token"));
        else if (a == "--short-read") o.short_read = std::max(0LL, std::atoll(next("--short-read")));
        else if (a == "--suffix-draft") o.suffix_draft = std::max(0, std::atoi(next("--suffix-draft")));
        else if (a == "--mtp-max-t") o.mtp_max_t = std::max(0, std::atoi(next("--mtp-max-t")));
        else if (a == "--lookup-chain") o.lookup_chain = std::clamp((int) std::strtol(next("--lookup-chain"), nullptr, 10), 0, 7);
        else if (a == "--lookup-chain-min") o.lookup_chain_min = std::max(3, (int) std::strtol(next("--lookup-chain-min"), nullptr, 10));
        else if (a == "--mtp-hnorm") {
            const std::string v = next("--mtp-hnorm");
            if (v != "pooled" && v != "stream") {
                std::fprintf(stderr, "strata generate: --mtp-hnorm takes pooled or stream\n");
                return 2;
            }
            o.mtp_hnorm_stream = v == "stream";
        }
        else if (a == "--control-vector") o.cvec_files.push_back({next("--control-vector"), 1.0f});
        else if (a == "--control-vector-scaled") {
            // FILE:SCALE, comma-separated; the last colon separates the scale from the path
            std::stringstream list(next("--control-vector-scaled"));
            std::string item;
            while (std::getline(list, item, ',')) {
                const size_t colon = item.rfind(':');
                char* end = nullptr;
                const float sc = colon == std::string::npos ? 0.0f : std::strtof(item.c_str() + colon + 1, &end);
                if (colon == std::string::npos || colon == 0 || end == item.c_str() + colon + 1 || *end != '\0') {
                    std::fprintf(stderr, "--control-vector-scaled: expected FILE:SCALE, got '%s'\n", item.c_str());
                    return 2;
                }
                o.cvec_files.push_back({item.substr(0, colon), sc});
            }
        }
        else if (a == "--control-vector-layer-range") {
            o.cvec_first = std::atoi(next("--control-vector-layer-range"));
            o.cvec_last = std::atoi(next("--control-vector-layer-range"));
        }
        else if (a == "--cvec-mode") {
            const std::string m = next("--cvec-mode");
            if (m == "project") o.cvec_mode = 0;
            else if (m == "add") o.cvec_mode = 1;
            else { std::fprintf(stderr, "--cvec-mode: add or project, got '%s'\n", m.c_str()); return 2; }
        }
        else if (a == "--cvec-dir") {
            const std::string d = next("--cvec-dir");
            if (d == "per-layer") o.cvec_single = -1;
            else if (d.rfind("single:", 0) == 0) o.cvec_single = std::atoi(d.c_str() + 7);
            else { std::fprintf(stderr, "--cvec-dir: per-layer or single:L, got '%s'\n", d.c_str()); return 2; }
        }
        else if (a == "--no-spec-split") o.spec_split = false;
        else if (a == "--eos-ids") {
            std::string e;
            if (!parse_i64_list(next("--eos-ids"), o.eos_ids, e)) { std::fprintf(stderr, "--eos-ids: %s\n", e.c_str()); return 2; }
            o.stop_eos = true;
        }
        else if (a == "--adapt-swaps") o.adapt_swaps = std::atoi(next("--adapt-swaps"));
        else if (a == "--expert-cache-cpu-order") o.expert_cache_cpu_order = true;
        else if (a == "--expert-cache-per-layer") o.expert_cache_per_layer = true;
        else if (a == "--no-hit-poke") o.no_hit_poke = true;
        else if (a == "--expert-profile") o.expert_profile = next("--expert-profile");
        else if (a == "--expert-profile-save") o.expert_profile_save = next("--expert-profile-save");
        else if (a == "--expert-profile-save-every")
            o.expert_profile_save_min = std::strtod(next("--expert-profile-save-every"), nullptr);
        else if (a == "--gpu-stages") o.gpu_stages = true;
        else if (a == "--coupled-draft") o.coupled_draft = true;
        else if (a == "--no-coupled-draft") o.coupled_draft = false;
        else if (a == "--shared-expert-arena") o.shared_expert_arena = next("--shared-expert-arena");
        else if (a == "--resident-cpu-experts") { o.mmap_experts = o.resident_cpu_experts = true; o.resident_gib = -1; }
        else if (a == "--mmap-experts") o.mmap_experts = true;
        else if (a == "--resident-experts") { o.mmap_experts = true; o.resident_gib = -1; }
        else if (a == "--resident-budget-gib") {
            o.mmap_experts = true;
            o.resident_gib = std::strtod(next("--resident-budget-gib"), nullptr);
            if (!(o.resident_gib > 0)) {
                std::fprintf(stderr, "strata generate: --resident-budget-gib takes a positive number of GiB\n");
                return 2;
            }
        }
        else if (a == "--stats") o.stats = true;
        else if (a == "--shared-late") o.shared_late = true;
        else if (a == "--keep-canonical") o.keep_canonical = true;
        else if (a == "--no-token-graph") o.no_token_graph = true;
        else if (a == "--no-fused-gr") o.no_fused_gr = true;
        else if (a == "--no-fast-attn") o.no_fast_attn = true;
        else if (a == "--no-publish-kernel") o.no_publish_kernel = true;
        else if (a == "--no-fused-gdn") o.no_fused_gdn = true;
        else if (a == "--no-fast-select") o.no_fast_select = true;
        else {
            // An unknown flag is an ERROR and not a warning: a typo'd `--max-neww` that silently generated 16
            // tokens would look like a working run.
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            usage();
            return 2;
        }
    }
    // Layer split (multi-GPU): the later stages run layers [K_i, K_i+1) on their own GPUs (--split-device, default
    // the next discrete ones); "auto" chooses the K before anything is loaded (below).  The helper caches
    // (--expert-cache-remote) are not carried by the Xe engine at all (remote_experts.cpp).
    if (o.serve && o.conversation_cache_mib > 0 && (o.prompt_cache == 0 || o.conversation_cache_slots == 0))
        std::fprintf(stderr, "strata serve: warning: conversation caching is disabled by %s\n",
                     o.prompt_cache == 0 ? "--prompt-cache 0" : "--conversation-cache-slots 0");
    if (o.conversation_save_compress && !strata::core::conversation_disk_can_compress()) {
        std::fprintf(stderr, "strata: --conversation-save-compress: this build has no c-blosc2 (install libblosc2-dev "
                             "and build again)\n");
        return 2;
    }
    // the folder keeps what the RAM cache parks: without parking there is nothing to write
    if (!o.conversation_save.empty() &&
        (o.conversation_cache_mib == 0 || o.conversation_cache_slots == 0 || o.prompt_cache == 0)) {
        if (o.serve)
            std::fprintf(stderr, "strata serve: warning: --conversation-save needs conversation parking "
                                 "(--conversation-cache-mib); not saving conversations\n");
        o.conversation_save.clear();
    }
    const bool pcie_given = o.pcie_frac >= 0.0;
    std::vector<int64_t> split_at;
    std::vector<int> split_devs;
    bool split_auto = false, split_same = false;
    if (!o.layer_split.empty()) {
        const int n_dev = strata::core::Runtime::count();
        auto ints = [](const std::string& str, auto& out) -> bool {
            using V = typename std::decay_t<decltype(out)>::value_type;
            size_t a = 0;
            while (a < str.size()) {
                size_t b = str.find(',', a);
                if (b == std::string::npos) b = str.size();
                const std::string t = str.substr(a, b - a);
                if (t.empty() || t.find_first_not_of("0123456789") != std::string::npos) return false;
                out.push_back((V) std::atoll(t.c_str()));
                a = b + 1;
            }
            return !out.empty();
        };
        split_auto = o.layer_split == "auto";
        bool ok = o.serve && (split_auto || ints(o.layer_split, split_at));
        // --split-device: engine numbers, or PCI addresses (setup writes those: its numbers are not the engine's)
        auto devices = [n_dev](const std::string& str, std::vector<int>& out) -> bool {
            size_t a = 0;
            while (a < str.size()) {
                size_t b = str.find(',', a);
                if (b == std::string::npos) b = str.size();
                std::string t = str.substr(a, b - a);
                if (t.empty()) return false;
                if (t.find_first_not_of("0123456789") == std::string::npos) {
                    out.push_back((int) std::strtol(t.c_str(), nullptr, 10));
                } else {
                    for (auto& c : t) c = (char) std::tolower((unsigned char) c);
                    int found = -1;
                    for (int d = 0; d < n_dev && found < 0; ++d)
                        if (strata::core::device_info(d).pci == t) found = d;
                    if (found < 0) {
                        std::fprintf(stderr, "strata generate: --split-device: no usable GPU at PCI address %s\n",
                                     t.c_str());
                        return false;
                    }
                    out.push_back(found);
                }
                a = b + 1;
            }
            return !out.empty();
        };
        if (ok && !o.split_device.empty()) ok = devices(o.split_device, split_devs);
        else if (ok)   // without --split-device: the other discrete GPUs (the processor's own graphics only by name)
            for (int d = 1; d < n_dev && (split_auto || split_devs.size() < split_at.size()); ++d)
                if (!strata::core::device_info(d).integrated) split_devs.push_back(d);
        if (ok && !split_auto && split_devs.empty() && split_at.size() == 1) split_devs.push_back(0);   // one GPU
        split_same = ok && split_devs.size() == 1 && split_devs[0] == 0 && !split_auto;
        if (ok && split_auto && split_devs.empty()) {
            std::fprintf(stderr, "strata generate: --layer-split auto: one GPU visible, so no split\n");
            o.layer_split.clear();
            split_auto = false;
        } else if (ok) {
            ok = (split_auto || split_at.size() == split_devs.size()) && split_devs.size() < (size_t) SplitDrive::kMax;
            for (size_t i = 0; ok && i < split_at.size(); ++i) ok = split_at[i] >= 2 && (i == 0 || split_at[i] > split_at[i - 1]);
            for (size_t i = 0; ok && !split_same && i < split_devs.size(); ++i) {
                ok = split_devs[i] > 0 && split_devs[i] < n_dev;
                for (size_t j = 0; ok && j < i; ++j) ok = split_devs[i] != split_devs[j];
            }
        }
        if (!ok) {
            std::fprintf(stderr, "strata generate: --layer-split K[,K2..]|auto needs --serve, rising K from 2, and one "
                                 "distinct GPU per K in --split-device (1..%d; or 0 with one K: the same GPU)\n", n_dev - 1);
            return 2;
        }
    }
    strata::core::set_coupled_draft(o.coupled_draft);
    if (o.mmap_experts && !o.shared_expert_arena.empty()) {
        std::fprintf(stderr, "strata generate: --shared-expert-arena cannot use --mmap-experts\n"); return 2;
    }
    if (o.resident_cpu_experts && (!o.mmap_experts || o.expert_profile.empty() || !split_devs.empty())) {
        std::fprintf(stderr, "strata generate: --resident-cpu-experts needs --mmap-experts and a static --expert-profile\n");
        return 2;
    }
    const bool multi_gpu = !split_devs.empty() && !split_same;
    // the helper-GPU expert caches (--expert-cache-remote, upstream's docs/SECOND_GPU.md): CUDA1..3 on one GPU; with a layer
    // split, the visible GPUs no stage runs on, in order
    int remote_dev[3] = {1, 2, 3};
    if (multi_gpu) {
        if (o.expert_profile.empty()) {
            std::fprintf(stderr, "strata generate: a layer split across GPUs needs --expert-profile\n");
            return 2;
        }
        int n_vis = 1;
        // one Xe device (docs/XE.md): no helper GPU is visible
        int next_free = 1;
        for (int r = 0; r < 3; ++r) {
            if (o.expert_cache_remote[(size_t) r] <= 0) continue;
            while (next_free < n_vis &&
                   std::find(split_devs.begin(), split_devs.end(), next_free) != split_devs.end()) ++next_free;
            if (next_free >= n_vis) {
                std::fprintf(stderr, "strata generate: --expert-cache-remote with a layer split needs a GPU that runs no "
                                     "stage (%d visible, %zu used by the split)\n", n_vis, split_devs.size() + 1);
                return 2;
            }
            remote_dev[r] = next_free++;
        }
        std::string devs;
        for (const int d : split_devs) devs += (devs.empty() ? "" : ",") + std::to_string(d);
        std::fprintf(stderr, "strata generate: layer split across %zu GPUs: CUDA0, then CUDA%s (split %s)\n",
                     split_devs.size() + 1, devs.c_str(), o.layer_split.c_str());
    }
    if (o.prefill_auto && (o.no_prefill_borrow || o.expert_profile.empty())) {
        o.prefill_auto = false;       // nothing to lend from: the buffers are reserved for the session, so keep them small
        o.prefill_chunk = 2048;
    }
    if (!have_tokens && o.serve) {   // plan v0.3 P8: requests bring their own tokens
        o.tokens = {248045};
        o.max_new = 1;
        have_tokens = true;
        o.stop_eos = true;
    }
    if (!have_tokens) {
        std::fprintf(stderr, "strata generate: --tokens is required (this build has no tokenizer; see the "
                             "header of src/program/generate.cpp)\n");
        usage();
        return 2;
    }

    if ((o.ple_io != "direct" && o.ple_io != "mmap" && o.ple_io != "ram") || o.ple_row_cache < 0 || o.ple_inflight < 1 ||
        o.ple_inflight > 1024 || !(o.ple_delay_us >= 0)) {
        std::fprintf(stderr, "strata generate: invalid --ple-io/--ple-row-cache/--ple-inflight/--ple-delay-us\n");
        return 2;
    }
    if (o.kv == "q4") o.kv = "q4_0";
    if (o.kv != "fp16" && o.kv != "int8" && o.kv != "q4_0" && o.kv != "k8v4") {
        std::fprintf(stderr, "strata generate: --kv must be fp16, int8, q4_0 or k8v4\n");
        return 2;
    }
    strata::core::qsa_set_kv_int8(o.kv == "int8");
    strata::core::qsa_set_kv_q4(o.kv == "q4_0");   // PR #21: 4-bit codes after a Hadamard rotation (kv_q4.hpp)
    // STRATA_KV_ROT=1: INT8 K/V through the Hadamard rotation --kv q4_0 already uses (upstream 270650e).  Opt-in:
    // upstream measured the first token's KL to FP16 K/V better on an NVFP4 pack (0.0066 -> 0.0051), worse on
    // IQ2_XS (0.0022 -> 0.0054)
    const char* kv_rot = std::getenv("STRATA_KV_ROT");
    strata::core::qsa_set_kv_int8_rotate(kv_rot != nullptr && kv_rot[0] == '1');
    strata::core::qsa_set_kv_hybrid(o.kv == "k8v4");   // K8V4: INT8 K + rotated Q4_0 V (upstream 2aa8f72)
    if (o.kv_resident < 0) {
        std::fprintf(stderr, "strata generate: --kv-resident must be >= 0\n");
        return 2;
    }
    strata::core::qsa_set_kv_resident(o.kv_resident);
    // Prompt lookup (the suffix drafter, on by default): the MTP keeps its --spec windows and a lookup window may be
    // up to 2 tokens longer; the draft policy (strata/spec/draft_policy.hpp) takes one only where it pays. Code
    // edits +6-11%, ordinary text unchanged (bench/results/2026-09-27-spec). --suffix-draft 0 turns it off.
    if (o.suffix_draft > 0 && o.spec >= 2 && o.mtp_max_t == 0) {
        o.mtp_max_t = o.spec;
        o.spec = std::min(o.spec + 2, 8);   // kVerifyMaxT
    }
    // --lookup-chain (opt-in): the MTP keeps its windows; a lookup chained after its drafts may lengthen one by up to K
    if (o.lookup_chain > 0 && o.spec >= 2) {
        if (o.mtp_max_t == 0) o.mtp_max_t = o.spec;
        o.spec = std::max(o.spec, std::min(o.mtp_max_t + o.lookup_chain, 8));   // kVerifyMaxT
    }
    strata::core::layer_set_shared_early(!o.shared_late);
    if (!o.native_preset.empty()) {
        // every shard of --native's model, once: a missing shard is an error that names it (upstream 02cfe36; the
        // shards used to be skipped silently when one could not be opened)
        std::vector<std::string> native_shards;
        try {
            native_shards = strata::gguf_split_paths(o.native_preset);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata generate: %s\n", e.what());
            return 2;
        }
        // --ple-gguf defaults to the shard that holds the PLE table (shard 2 of the ISTA files and of UD-Q4_K_XL)
        if (!o.no_ple && o.ple_gguf.empty()) {
            try {
                const strata::GgufModel model(native_shards);
                size_t at = 0;
                if (model.find("per_layer_token_embd.weight", &at)) o.ple_gguf = model.shard(at).path();
            } catch (const std::exception& e) {
                std::fprintf(stderr, "strata generate: %s\n", e.what());
                return 2;
            }
        }
        if (o.no_ple || o.ple_gguf.empty()) {
            std::fprintf(stderr, "strata generate: --native requires --ple-gguf (the PLE key is native too)\n");
            return 2;
        }
        o.stream_token = true;
        o.gr_native_mmvf = true;
        o.native_bf16 = o.native_bf16_extra = true;
        o.native_ple_key = o.native_moe_combine = o.native_gdn = o.native_router = true;
        o.native_qsa = o.native_qsa_indexer = o.native_rope = o.native_ple_postops = true;
        if (o.native_head_gguf.empty()) o.native_head_gguf = o.native_preset;
        if (o.native_dense_gguf.empty()) {
            // every shard of the model (<name>-0000N-of-0000M.gguf beside --native), then the PLE shard: a split
            // may put any layer in any shard (Swift's GGUFs: layers 13-47 in shard 2, the PLE table in shard 1)
            o.native_dense_gguf = native_shards;
            // a PLE-only table (tools/ple_fp8_pack.py: architecture strata-ple) holds no projections
            bool ple_only = false;
            try {
                strata::GgufFile pg(o.ple_gguf);
                if (const strata::MetaValue* v = pg.get("general.architecture")) ple_only = v->s == "strata-ple";
            } catch (const std::exception&) {
                ple_only = false;   // an unreadable table fails with its own message when it loads
            }
            if (!ple_only &&
                std::find(o.native_dense_gguf.begin(), o.native_dense_gguf.end(), o.ple_gguf) == o.native_dense_gguf.end())
                o.native_dense_gguf.push_back(o.ple_gguf);
        }
        // Plan v0.3 (24 Sep): the CPU experts stay on the VNNI kernel.  The llama.cpp-CPU-exact q8_0 contract
        // cost 27.0 vs 17.2 ms/token of pool time and G-C does not need it; `--cpu-oracle-q8-0` still selects it.
    }
    if (o.logits_stride > 1 && (o.max_new != 1 || o.dump_logits.empty())) {
        std::fprintf(stderr, "strata generate: --logits-stride > 1 requires --max-new 1 and --dump-logits\n");
        return 2;
    }
    if (o.no_ple && !o.ple_gguf.empty()) {
        std::fprintf(stderr, "strata generate: --no-ple and --ple-gguf are mutually exclusive\n");
        return 2;
    }
    if (o.native_ple_postops && o.no_ple) {
        std::fprintf(stderr, "strata generate: --native-ple-postops requires PLE enabled\n");
        return 2;
    }
    if (!o.no_ple && o.ple_gguf.empty()) {
        std::fprintf(stderr, "strata generate: --ple-gguf is required; --no-ple explicitly enables a diagnostic ablation\n");
        return 2;
    }
    // P7 audit: positions, cells and pooled-block indices are cast to int32 on the device path.
    if (o.max_context > 2147483647LL - 8) {
        std::fprintf(stderr, "strata generate: --max-context must be below 2^31\n");
        return 2;
    }
    if (o.max_new <= 0 || o.max_context <= 0 || o.max_new > o.max_context ||
        o.tokens.size() > (size_t) (o.max_context - o.max_new)) {
        std::fprintf(stderr, "strata generate: positive --max-new and --max-context must fit the prompt and generation\n");
        return 2;
    }
    // The rope knobs' ranges, at second zero; the configuration itself resolves once the model file has had its say
    strata::kernels::RopeScaling rope_cfg;
    {
        using RST = strata::kernels::RopeScalingType;
        if (o.rope_scaling == "none") rope_cfg.type = RST::None;
        else if (o.rope_scaling == "linear") rope_cfg.type = RST::Linear;
        else if (o.rope_scaling == "yarn") rope_cfg.type = RST::YaRN;
        else if (!o.rope_scaling.empty()) {
            std::fprintf(stderr, "strata generate: --rope-scaling must be none, linear or yarn (got '%s')\n",
                         o.rope_scaling.c_str());
            return 2;
        }
        // finite first: atof("nan") passes every range comparison below
        for (const double v : {o.rope_scale, o.rope_freq_base, o.rope_freq_scale, o.yarn_orig_ctx, o.yarn_ext_factor,
                               o.yarn_attn_factor, o.yarn_beta_fast, o.yarn_beta_slow})
            if (!std::isfinite(v)) {
                std::fprintf(stderr, "strata generate: a rope scaling knob is not a finite number (%g)\n", v);
                return 2;
            }
        if (o.rope_scale != 0 && o.rope_scale < 1.0) {
            std::fprintf(stderr, "strata generate: --rope-scale %g must be >= 1 (it extends the context)\n", o.rope_scale);
            return 2;
        }
        if (o.rope_freq_base != 0 && o.rope_freq_base <= 1.0) {
            std::fprintf(stderr, "strata generate: --rope-freq-base must be a base above 1 (0 = the model's)\n");
            return 2;
        }
        if (o.rope_freq_scale < 0 || o.yarn_orig_ctx < 0 || o.yarn_ext_factor < -1.0 || o.yarn_attn_factor <= 0 ||
            o.yarn_beta_fast <= 0 || o.yarn_beta_slow <= 0) {
            std::fprintf(stderr, "strata generate: invalid rope scaling knob (--yarn-ext-factor < 0 = auto, "
                                 "--yarn-orig-ctx 0 = default, the rest positive)\n");
            return 2;
        }
    }
    if (!std::isfinite(o.temperature) || o.temperature < 0 || !std::isfinite(o.top_p) ||
        o.top_p <= 0 || o.top_p > 1 || o.top_k < 0 || o.expert_cache < -1 ||
        o.pool_workers < 0 || std::any_of(o.expert_cache_remote.begin(), o.expert_cache_remote.end(),
                                           [](int slots) { return slots < 0; }) ||
        (o.expert_cache_remote[1] > 0 && o.expert_cache_remote[0] == 0) ||
        (o.expert_cache_remote[2] > 0 && o.expert_cache_remote[1] == 0)) {
        std::fprintf(stderr, "strata generate: invalid sampling or resource parameter\n");
        return 2;
    }
    if (o.expert_cache_remote_placement != "stripe" && o.expert_cache_remote_placement != "layer") {
        std::fprintf(stderr, "strata generate: --expert-cache-remote-placement must be stripe or layer\n");
        return 2;
    }

    if (o.native_flash_attn_short && o.max_context > 256) {
        std::fprintf(stderr, "strata generate: --native-flash-attn-short requires --max-context <=256\n");
        return 2;
    }
    if (o.native_flash_attn_short && (o.gpu_only_full || o.graph_only || o.gpu_stages)) {
        std::fprintf(stderr, "strata generate: --native-flash-attn-short requires the normal decode loop for status validation\n");
        return 2;
    }
    if (o.native_ple_key && (o.native_dense_gguf.empty() || o.no_ple)) {
        std::fprintf(stderr, "strata generate: --native-ple-key requires PLE and --native-dense-gguf\n");
        return 2;
    }
    if (o.cpu_oracle_q8_0 && (o.expert_cache != 0 || !o.expert_profile.empty())) {
        std::fprintf(stderr, "strata generate: --cpu-oracle-q8-0 cannot be combined with --expert-cache or --expert-profile until the GPU expert contract matches\n");
        return 2;
    }

    // **BEFORE ANYTHING ELSE.**  The CPU expert kernel is AVX-512 (VNNI + VBMI) and its translation unit is
    // compiled with AVX-512 enabled, so on a CPU without those features it does not fail - it executes an illegal
    // instruction at some unpredictable token.  Refusing at second zero is the whole point of P2.S3's check.
    strata::kernels::cpu::expert_set_oracle_q8_0(o.cpu_oracle_q8_0);

    std::string err;
    if (!o.native_head_gguf.empty() && !o.stream_token) {
        std::fprintf(stderr, "--native-head-gguf requires --stream-token\n");
        return 2;
    }
    // Plan v0.3 P6: where the experts live.  A native pack (tools/iq_pack.py: the IQ2_XS / IQ3_XXS files) keeps
    // every quantized tensor in its GGUF form, so it needs --native (the dense projections, head and embedding
    // come from the model file) and runs its experts in verify windows only (--spec).
    {
        const strata::core::ModelGeometry g0;
        if (!strata::kernels::cpu::expert_layout_load(o.pack, g0.n_layers, g0.n_expert, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        // Every layer's formats must have GPU expert kernels and a prompt-path dequantizer, checked here, before
        // anything is allocated (upstream efeffd8): an unsupported type would only fail inside the first verify window.
        const auto& lay = strata::kernels::cpu::expert_layout();
        for (int64_t l = 0; lay.native && l < (int64_t) lay.fmt.size(); ++l) {
            const auto& f = lay.fmt[(size_t) l];
            if (!strata::kernels::native_expert_supported(f.gu_type, f.d_type, f.n_embd, f.n_ff)) {
                std::fprintf(stderr, "strata generate: layer %lld's experts are %s/%s (ggml types %d/%d), which this "
                                     "engine has no GPU kernels for\n", (long long) l,
                             strata::ggml_type_name((uint32_t) f.gu_type), strata::ggml_type_name((uint32_t) f.d_type),
                             f.gu_type, f.d_type);
                return 1;
            }
        }
    }
    const bool native_pack = strata::kernels::cpu::expert_layout().native;
    if (o.expert_cache_remote[0] > 0) {
        // Keep CUDA1's proven startup order: initialise its context before
        // allocating GPU0 weights or mapping the large host expert arena.
        double free_gib = 0;
        if (!strata::core::RemoteExperts::preflight(remote_dev[0], free_gib, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: CUDA%d context ready, %.2f GiB free before expert arena registration\n",
                     remote_dev[0], free_gib);
    }
    strata::core::Runtime::get().preload_kernels();   // beside the model's loading
    // and on every other GPU of a layer split, with its matrix products' library (each GPU builds its own kernels)
    if (multi_gpu)
        for (const int d : split_devs) strata::core::Runtime::at(d).preload_kernels();
    strata::prefill::Gemm::prepare();
    if (multi_gpu)
        for (const int d : split_devs) {
            const strata::core::OnDevice on(d);
            strata::prefill::Gemm::prepare();
        }
    {
        // the GPU this run drives and the paths it takes (chosen from what it reports, never from its name)
        const strata::core::DeviceInfo gi = strata::core::device_info(0);
        std::fprintf(stderr, "strata generate: GPU %s, %u compute units, %.1f GiB, prompt matrix products on %s\n",
                     gi.name.c_str(), gi.compute_units, (double) gi.total_bytes / (1024.0 * 1024 * 1024),
                     strata::prefill::Gemm::path());
    }
    // plan v0.3 P6: the PCIe share of the missed experts, measured per kind of pack (the paper, finding on PCIe).
    // PR #44: a x8 link carries half of what the native default assumes - the GPU's SMs read that share over the
    // link (the copy kernel, since 0.1.14), so on a slower link it must shrink or the window waits for it.  The
    // real H2D bandwidth is probed once; from 20 GB/s up (x16 PCIe 4/5) the measured default stays.  The canonical
    // pack's 0.2 was never measured against the link, so it is left alone.  `--calibrate` measures it outright.
    if (o.pcie_frac < 0.0) {
        const double base = native_pack ? 0.55 : 0.2;
        std::string bursts;
        const double bw = native_pack ? probe_pcie_h2d_gbps(&bursts) : -1.0;
        if (!native_pack) {
            o.pcie_frac = base;
        } else if (bw > 0.0) {
            o.pcie_frac = pcie_frac_for_gbps(bw, base);
            std::fprintf(stderr, "strata generate: PCIe probe: %.1f GB/s host->device (best of %s) -> pcie_frac %.2f "
                                 "(default %.2f)\n", bw, bursts.c_str(), o.pcie_frac, base);
        } else {
            o.pcie_frac = base;
            std::fprintf(stderr, "strata generate: PCIe probe failed -> pcie_frac default %.2f\n", base);
        }
    }
    // the canonical Q2_0 pack's CPU kernels are AVX-512 only; a native pack runs on AVX2 CPUs as well
    if (!native_pack) strata::kernels::cpu::cpu_require_expert_support();
    else if (!strata::kernels::cpu::cpu_avx512_ok())
        std::fprintf(stderr, "strata generate: this CPU has no AVX-512: the expert kernels run on %s "
                             "(multi-token for the i-quant gate/up rows)\n",
                     std::getenv("STRATA_NO_IQ256") == nullptr ? "AVX-2" : "ggml-cpu vec_dot (STRATA_NO_IQ256 set)");
    strata::core::NativeEmbed native_embed;
    if (native_pack) {
        if (o.native_preset.empty() || o.spec < 2 || o.keep_canonical ||
            (o.prefill_chunk <= 0 && o.tokens.size() > 1)) {
            std::fprintf(stderr, "strata generate: %s is a native (IQ) pack: it needs --native SHARD1, --spec T (T >= 2) "
                                 "and --prefill CHUNK\n", o.pack.c_str());
            return 2;
        }
        const strata::core::ModelGeometry g0;
        const auto embed_t0 = std::chrono::steady_clock::now();
        if (!native_embed.load(o.embd_gguf.empty() ? o.native_preset : o.embd_gguf, g0.n_embd, 248320, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        strata::core::set_native_embed(&native_embed);
        std::fprintf(stderr, "strata generate: native pack: %s experts (largest blob %.2f MB), token embedding "
                             "%s in mapped host memory (%.0f MiB, %.1f s)\n",
                     o.pack.c_str(), (double) strata::kernels::cpu::expert_layout().max_blob / 1e6,
                     strata::ggml_type_name((uint32_t) native_embed.type()), (double) native_embed.bytes() / 1048576.0,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - embed_t0).count());
    }
    // Plan v0.3 P1: tensors served in native form are not also loaded in canonical form (~2.7 GB of VRAM back
    // to the expert cache with --native).  `--keep-canonical` loads both, as before.
    std::set<std::string> skip;
    if (!o.keep_canonical) {
        if (!o.native_dense_gguf.empty() &&
            !strata::core::NativeDense::served_names(o.native_dense_gguf, o.native_ple_key, skip, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        if (!o.native_head_gguf.empty()) skip.insert("output.weight");
        // the PLE module validates its canonical key at construction (8 MB); a native pack has none to load
        if (!native_pack) skip.erase("blk.1.ple_key.weight");
        else if (!strata::core::NativeDense::keep_unquantized_ple_key(o.pack, skip, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str()); return 1;
        }
        if (native_pack) skip.insert("token_embd.weight");
    }
    // ---- --layer-split auto: the split points, chosen before anything is loaded - from what this PC measures and
    // the sizes the files and the geometry give - so the placement then loads as an explicit one: each GPU's sessions
    // carved to its layers, and its dense weights too.
    //   The window time of a placement: per layer, the dense weights read at the GPU's own bandwidth, then the
    //   layer's routed experts - the cached share read by the GPU at its bandwidth while the CPU reads the rest at the
    //   RAM's (the two overlap: the larger counts) - and a hand-off per stage boundary.
    //   Which experts a cache holds: its layers' profiled pairs, hottest first, until the VRAM its GPU has left is
    //   used: free now, less the reserve, the weights, the sessions (--batch slots too), the prompt path and, on the
    //   last GPU, the head; a later GPU keeps 1 GiB more (its windows and the drafter; and what KV streaming's staging
    //   would take, which is not counted here).  A window routes about k x rows experts a layer (rows:
    //   STRATA_SPLIT_ROWS, default 3, the average verify window with MTP drafts).
    if (multi_gpu && split_auto) {
        // the geometry, as set up below (the canonical one, the model file's MoE shape)
        strata::core::ModelGeometry ga;
        int64_t Ka = 10;
        if (!o.native_preset.empty()) {
            try {
                strata::GgufFile model_gguf(o.native_preset);
                if (const strata::MetaValue* v = model_gguf.get("qwen4exp.expert_count")) ga.n_expert = (int64_t) v->u;
                if (const strata::MetaValue* v = model_gguf.get("qwen4exp.expert_used_count")) Ka = (int64_t) v->u;
            } catch (const std::exception& e) {
                std::fprintf(stderr, "strata generate: layer split auto: %s (sizes from the default geometry)\n", e.what());
            }
        }
        const auto& lay = strata::kernels::cpu::expert_layout();
        const int ns = (int) split_devs.size() + 1;
        const int64_t L = ga.n_layers;
        std::vector<std::pair<int32_t, int32_t>> prof;
        int64_t pslots = 0;
        if (!strata::core::read_expert_profile(o.expert_profile, ga.n_layers, ga.n_expert, prof, pslots, err)) {
            std::fprintf(stderr, "strata generate: --layer-split auto: %s\n", err.c_str());
            return 1;
        }
        auto dev_of_stage = [&](int i) { return i == 0 ? 0 : split_devs[(size_t) i - 1]; };
        // the weights a range holds: the canonical arena's and the native projections'
        uint64_t canon_all = 0;
        if (!strata::core::WeightTable::pool_bytes(o.pack, canon_all, err, skip.empty() ? nullptr : &skip)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::vector<uint64_t> canon_layer((size_t) L, 0);
        {   // each layer's share: the arena without that layer's blk.N.* tensors (PLE ones excepted)
            std::vector<std::vector<std::string>> names((size_t) L);
            if (std::FILE* f = std::fopen((o.pack + "/index.txt").c_str(), "rb")) {
                char line[1024], name[256];
                while (std::fgets(line, sizeof line, f)) {
                    if (line[0] == '#' || std::sscanf(line, "%255s", name) != 1) continue;
                    const std::string nm = name;
                    if (nm.rfind("blk.", 0) != 0 || nm.find("ple") != std::string::npos) continue;
                    const long l = std::strtol(name + 4, nullptr, 10);
                    if (l >= 0 && l < L) names[(size_t) l].push_back(nm);
                }
                std::fclose(f);
            }
            for (int64_t l = 0; l < L; ++l) {
                std::set<std::string> without = skip;
                without.insert(names[(size_t) l].begin(), names[(size_t) l].end());
                uint64_t b = 0;
                if (strata::core::WeightTable::pool_bytes(o.pack, b, err, &without) && b <= canon_all)
                    canon_layer[(size_t) l] = canon_all - b;
            }
        }
        std::vector<uint64_t> nat_layer;
        uint64_t nat_shared = 0, nat_all = 0;
        if (!o.native_dense_gguf.empty() &&
            !strata::core::NativeDense::layer_bytes(o.native_dense_gguf, o.native_ple_key, L, nat_layer, nat_shared, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        nat_layer.resize((size_t) L, 0);
        nat_all = nat_shared;
        for (const uint64_t b : nat_layer) nat_all += b;
        uint64_t head_bytes = 0;   // the native head, on the last GPU
        if (!o.native_head_gguf.empty()) {
            try {
                const strata::GgufModel hm = strata::GgufModel::open(o.native_head_gguf);
                size_t at = 0;
                if (const strata::TensorInfo* t = hm.find("output.weight", &at)) head_bytes = strata::tensor_payload_bytes(*t);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "strata generate: layer split auto: %s (the head left out of the sizes)\n", e.what());
            }
        }
        auto weights_of = [&](int64_t lo, int64_t hi) -> uint64_t {
            uint64_t b = canon_all + nat_shared;
            for (int64_t l = 0; l < L; ++l)
                if (l < lo || l >= hi) b -= std::min(b, canon_layer[(size_t) l]);
            for (int64_t l = lo; l < hi; ++l) b += nat_layer[(size_t) l];
            return b;
        };
        // what each GPU has, and what its prompt path takes (the chunk's buffers and the streamed ring, on that GPU;
        // nothing when it borrows them from the expert cache, as it does with a profile)
        const bool own_prompt = o.no_prefill_borrow || o.expert_profile.empty();
        std::vector<int64_t> free0((size_t) ns), prompt((size_t) ns);
        strata::core::QsaState q_ctx{};
        q_ctx.max_cells = o.max_context;
        strata::core::SessionState s_ctx;
        s_ctx.qsa_states = &q_ctx;
        s_ctx.k = Ka;
        for (int i = 0; i < ns; ++i) {
            const strata::core::OnDevice on(dev_of_stage(i));
            size_t fb = 0, tb = 0;
            strata::gpu::mem_info(&fb, &tb);
            free0[(size_t) i] = (int64_t) fb;
            prompt[(size_t) i] = o.prefill_chunk > 0 && own_prompt
                                     ? (int64_t) strata::prefill::Prefill::bytes_needed(ga, s_ctx, o.prefill_chunk) : 0;
        }
        // the room for experts a stage has, before it is clamped at 0: below what the prompt path needs, the stage
        // cannot start (upstream 09e5ab4f)
        auto cap_raw = [&](int i, int64_t lo, int64_t hi, bool last) -> int64_t {
            const int64_t base_reserve =   // --vram-reserve-later-mib: the later cards' own reserve (upstream 5c4105c0)
                i > 0 && o.vram_reserve_later_mib >= 0 ? o.vram_reserve_later_mib : o.vram_reserve_mib;
            const int64_t reserve = (base_reserve + (i > 0 ? 1024 : 0)) << 20;
            const int64_t sess = (int64_t) strata::core::session_bytes(ga, o.max_context, Ka, lo, hi) *
                                 (1 + std::max(o.batch, 0));
            return free0[(size_t) i] - reserve - (int64_t) weights_of(lo, hi) - sess - prompt[(size_t) i] -
                   (last ? (int64_t) head_bytes : 0);
        };
        // what a stage needs at least for its prompt path: its own buffers are in prompt[] already; a stage that
        // borrows them lends cache slots for the last-resort chunk of 512 tokens
        const int64_t prompt_min = own_prompt || o.prefill_chunk <= 0
                                       ? 0 : (int64_t) strata::prefill::Prefill::bytes_needed(ga, s_ctx, 512);
        // each GPU reading its own memory (bytes per ms; random bytes: a memset's run of one value is compressed on
        // some GPUs, and the B70 then read it at over 1.7 TB/s)
        std::vector<double> bw((size_t) ns);
        auto gpu_bandwidth = [&](int dev) -> double {
            const strata::core::OnDevice on(dev);
            const size_t n = (size_t) 512 << 20;
            std::vector<uint64_t> host(n / 8);
            uint64_t x = 0x9e3779b97f4a7c15ull;
            for (uint64_t& v : host) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; v = x; }
            void* a = nullptr;
            void* b = nullptr;
            std::vector<double> got;
            if (strata::gpu::alloc_device(&a, n) && strata::gpu::alloc_device(&b, n) &&
                strata::gpu::copy(a, host.data(), n) && strata::gpu::copy(b, a, n)) {
                for (int rep = 0; rep < 5; ++rep) {
                    const Clock::time_point t0 = Clock::now();
                    if (!strata::gpu::copy(b, a, n)) break;
                    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                    got.push_back(2.0 * (double) n / std::max(ms, 1e-3));   // read + write
                }
            }
            if (a) strata::gpu::free(a);
            if (b) strata::gpu::free(b);
            if (got.empty()) return 100e6;   // (100 GB/s when the copy could not run)
            std::sort(got.begin(), got.end());
            return got[got.size() / 2];
        };
        // the RAM as the CPU expert pool reads it (one reader per hardware thread; STRATA_SPLIT_CPU_GBPS overrides)
        const double cpu_bw = [&] {
            if (const char* v = std::getenv("STRATA_SPLIT_CPU_GBPS")) return std::strtod(v, nullptr) * 1e6;
            const int nt = std::max(1, (int) std::thread::hardware_concurrency());
            const size_t per = (size_t) 32 << 20;
            std::vector<std::vector<uint64_t>> bufs((size_t) nt, std::vector<uint64_t>(per / 8, 1));
            double best = 0;
            for (int rep = 0; rep < 3; ++rep) {
                std::atomic<uint64_t> sink{0};
                std::vector<std::thread> th;
                th.reserve((size_t) nt);
                const Clock::time_point t0 = Clock::now();
                for (int t = 0; t < nt; ++t)
                    th.emplace_back([&, t] {
                        uint64_t acc = 0;
                        for (const uint64_t v : bufs[(size_t) t]) acc += v;
                        sink += acc;
                    });
                for (auto& t : th) t.join();
                const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                best = std::max(best, (double) nt * (double) per / std::max(ms, 1e-3));
            }
            return best;
        }();
        for (int i = 0; i < ns; ++i) {
            bw[(size_t) i] = gpu_bandwidth(dev_of_stage(i));
            std::fprintf(stderr, "strata generate: layer split auto: CUDA%d reads %.0f GB/s, %.2f GiB free\n",
                         dev_of_stage(i), bw[(size_t) i] / 1e6, (double) free0[(size_t) i] / 1073741824.0);
        }
        std::fprintf(stderr, "strata generate: layer split auto: the CPU experts read RAM at %.0f GB/s\n", cpu_bw / 1e6);
        const double rows = std::getenv("STRATA_SPLIT_ROWS") ? std::strtod(std::getenv("STRATA_SPLIT_ROWS"), nullptr) : 3.0;
        const double dense_layer = (double) (canon_all + nat_all) / (double) L;
        // the routed mass of rank r: (r+1)^-a.  a = 0.8 fits this model's measured hit rates on the B70 (IQ3_S: 58% of
        // the pairs cached, 91% hits; UD-Q4_K_XL: 34%, 78%); upstream's 1.2 (the Coder, 256 experts) put both near 98%.
        // A profile that held each pair's count would replace the fit.  STRATA_SPLIT_MASS_EXP overrides a.
        const double mass_exp = std::getenv("STRATA_SPLIT_MASS_EXP") ? std::strtod(std::getenv("STRATA_SPLIT_MASS_EXP"), nullptr) : 0.8;
        std::vector<double> mass(prof.size());
        for (size_t r = 0; r < prof.size(); ++r) mass[r] = std::pow((double) r + 1.0, -mass_exp);
        std::vector<double> layer_mass((size_t) L, 0.0), layer_held((size_t) L, 0.0);
        for (size_t r = 0; r < prof.size(); ++r) layer_mass[(size_t) prof[r].first] += mass[r];
        auto cost = [&](int64_t l) -> int64_t {
            return native_pack ? ((int64_t) lay.blob_bytes(l) + 255) / 256 * 256 : (int64_t) lay.max_blob;
        };
        const double hop_ms = 0.1;   // a hand-off between two GPUs (5 tokens: ~60 us measured, B70 -> RTX 4070)
        // the predicted window time (ms) of a placement (`at`: the split points; empty: CUDA0 alone)
        std::vector<int64_t> cap((size_t) ns), used((size_t) ns);
        bool startable = true;   // predict's: every stage can start its prompt path
        auto predict = [&](const std::vector<int64_t>& at, double& held_mass, int64_t& held) -> double {
            const int nst = (int) at.size() + 1;
            startable = true;
            for (int i = 0; i < nst; ++i) {
                const int64_t lo = i == 0 ? 0 : at[(size_t) i - 1], hi = i + 1 < nst ? at[(size_t) i] : L;
                const int64_t raw = cap_raw(i, nst == 1 ? 0 : lo, nst == 1 ? -1 : hi, i + 1 == nst);
                if (raw < prompt_min) startable = false;
                cap[(size_t) i] = std::max<int64_t>(raw, 0);
            }
            std::fill(used.begin(), used.end(), 0);
            std::fill(layer_held.begin(), layer_held.end(), 0.0);
            held = 0;
            std::vector<bool> full((size_t) ns, false);
            for (size_t r = 0; r < prof.size(); ++r) {
                const int64_t l = prof[r].first;
                int st = 0;
                while (st + 1 < nst && l >= at[(size_t) st]) ++st;
                if (full[(size_t) st]) continue;
                if (used[(size_t) st] + cost(l) > cap[(size_t) st]) { full[(size_t) st] = true; continue; }   // as the fill
                used[(size_t) st] += cost(l);
                layer_held[(size_t) l] += mass[r];
                ++held;
            }
            double ms = hop_ms * (double) (nst - 1), all = 0, kept = 0;
            for (int64_t l = 0; l < L; ++l) {
                int st = 0;
                while (st + 1 < nst && l >= at[(size_t) st]) ++st;
                const double hit = layer_mass[(size_t) l] > 0 ? layer_held[(size_t) l] / layer_mass[(size_t) l] : 1.0;
                const double experts = (double) Ka * rows * (double) lay.blob_bytes(l);
                ms += dense_layer / bw[(size_t) st] +
                      std::max(experts * hit / bw[(size_t) st], experts * (1.0 - hit) / cpu_bw);
                all += layer_mass[(size_t) l];
                kept += layer_held[(size_t) l];
            }
            held_mass = all > 0 ? kept / all : 1.0;
            return ms;
        };
        {
            double hm = 0;
            int64_t held = 0;
            const double one = predict({}, hm, held);
            std::fprintf(stderr, "strata generate: layer split auto: CUDA0 alone would take %.1f ms per window (~%.1f%% "
                                 "of the routed mass cached)\n", one, 100.0 * hm);
        }
        std::vector<int64_t> best, at((size_t) ns - 1);
        double best_ms = 1e30, best_mass = 0;
        int64_t best_held = 0, best_room = 0;
        bool gate = true;   // only placements whose every stage can start its prompt path (upstream 09e5ab4f)
        auto consider = [&]() {
            double hm = 0;
            int64_t held = 0;
            const double ms = predict(at, hm, held);
            if (gate && !startable) return;
            // the room the fullest stage keeps after its cache: on a tie the balanced placement wins (upstream 08cdd8bf)
            int64_t room = INT64_MAX;
            for (size_t i = 0; i < at.size() + 1; ++i) room = std::min(room, cap[i] - used[i]);
            if (ms < best_ms - 1e-9 || (ms <= best_ms + 1e-9 && room > best_room)) {
                best = at; best_ms = ms; best_mass = hm; best_held = held; best_room = room;
            }
        };
        auto search = [&] {
        if (ns == 2) {
            for (int64_t k = 2; k < L; ++k) { at[0] = k; consider(); }
        } else if (ns == 3) {
            for (int64_t k1 = 2; k1 + 1 < L; ++k1)
                for (int64_t k2 = k1 + 1; k2 < L; ++k2) { at[0] = k1; at[1] = k2; consider(); }
        } else if (ns == 4) {   // every four-way placement as well (upstream e2b32892): ~15,000 for 48 layers
            for (int64_t k1 = 2; k1 + 2 < L; ++k1)
                for (int64_t k2 = k1 + 1; k2 + 1 < L; ++k2)
                    for (int64_t k3 = k2 + 1; k3 < L; ++k3) { at[0] = k1; at[1] = k2; at[2] = k3; consider(); }
        } else {
            double total = 0;
            for (const double c : bw) total += c;
            double acc = 0;
            for (int i = 0; i + 1 < ns; ++i) {
                acc += bw[(size_t) i];
                at[(size_t) i] = std::clamp<int64_t>((int64_t) std::llround(acc / total * (double) L),
                                                    i == 0 ? 2 : at[(size_t) i - 1] + 1, L - (ns - 1 - i));
            }
            consider();
        }
        };
        search();
        if (best.empty()) {   // no placement can start on paper: the choice without the gate, and what is short
            gate = false;
            search();
            std::fprintf(stderr, "strata generate: layer split auto: WARNING: no placement leaves every card room for the "
                                 "prompt path's buffers (%lld MiB a stage%s); the engine may run out of VRAM at start: a "
                                 "smaller --max-context, a smaller --prefill chunk or a smaller --vram-reserve-mib leaves "
                                 "more\n",
                         (long long) (std::max<int64_t>(prompt_min, 0) >> 20),
                         own_prompt ? ", its own buffers counted in" : " lent from its cache");
        }
        split_at = best;
        split_auto = false;   // from here on, as an explicit --layer-split
        std::string ks;
        for (const int64_t k : split_at) ks += (ks.empty() ? "" : ",") + std::to_string(k);
        std::fprintf(stderr, "strata generate: layer split auto: K=%s - predicted %.1f ms per decode window; the caches "
                             "hold %lld of %zu profiled pairs (~%.1f%% of the routed mass)\n", ks.c_str(), best_ms,
                     (long long) best_held, prof.size(), 100.0 * best_mass);
    }
    // Layer split across GPUs (upstream e50f2663, 12c099b3, 13f70a34): every GPU holds only the dense weights of ITS
    // layers (the PLE tensors stay everywhere).  A full copy on each card (~1.5 GB for IQ2_XS) is VRAM the expert cache
    // wants: upstream's opt-in STRATA_STAGE_TRIM=1 is always on here (IQ2_XS, 8 GB cards: hits 22% -> 32%, 7% faster).
    // (Upstream efec5694 keeps CUDA0's routers for RouterLookahead, which the Xe engine does not have.)
    const std::set<std::string> skip_base = skip;
    const bool stage_trim = multi_gpu && !split_auto && !split_at.empty();
    auto add_foreign = [&](int64_t lb, int64_t le, std::set<std::string>& out) {
        std::FILE* f = std::fopen((o.pack + "/index.txt").c_str(), "rb");
        if (!f) return;
        char line[1024], name[256];
        while (std::fgets(line, sizeof line, f)) {
            if (line[0] == '#' || std::sscanf(line, "%255s", name) != 1) continue;
            const std::string n = name;
            if (n.rfind("blk.", 0) != 0 || n.find("ple") != std::string::npos) continue;
            const int64_t l = std::strtoll(name + 4, nullptr, 10);
            if (l < lb || l >= le) out.insert(n);
        }
        std::fclose(f);
    };
    if (stage_trim) {
        add_foreign(0, split_at[0], skip);
        strata::core::NativeDense::set_layer_range(0, (int) split_at[0]);
        std::fprintf(stderr, "strata generate: layer split: CUDA0 loads the dense weights of layers 0-%lld only\n",
                     (long long) split_at[0] - 1);
    }
    uint64_t pool_bytes = 0;
    if (!strata::core::WeightTable::pool_bytes(o.pack, pool_bytes, err, skip.empty() ? nullptr : &skip)) {
        std::fprintf(stderr, "strata generate: %s\n", err.c_str());
        return 1;
    }
    void* arena = nullptr;
    if (!strata::gpu::alloc_device(&arena, pool_bytes)) {
        // #486 (upstream bd649ac): the arena is the first large allocation and its size does not depend on the
        // context, so what is missing is held by something else: say how much was free
        size_t free_b = 0, total_b = 0;
        strata::gpu::mem_info(&free_b, &total_b);
        std::fprintf(stderr, "strata generate: device allocation of %llu bytes for the weight arena failed (%s): %llu "
                             "MiB of %llu MiB VRAM free on this GPU. The arena is allocated first, before the KV and "
                             "expert caches: another program (or an engine that is still exiting) holds the rest\n",
                     (unsigned long long) pool_bytes, strata::gpu::last_error(), (unsigned long long) (free_b >> 20),
                     (unsigned long long) (total_b >> 20));
        return 1;
    }
    auto load_t0 = std::chrono::steady_clock::now();
    auto load_s = [&] {
        const auto t = std::chrono::steady_clock::now();
        const double s = std::chrono::duration<double>(t - load_t0).count();
        load_t0 = t;
        return s;
    };
    strata::core::WeightTable wt;
    if (!wt.load(o.pack, arena, pool_bytes, err, skip.empty() ? nullptr : &skip)) {
        std::fprintf(stderr, "strata generate: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "strata generate: %llu MiB of weights loaded from %s in %.1f s (%zu canonical tensors "
                         "skipped: served natively)\n",
                 (unsigned long long) (pool_bytes >> 20), o.pack.c_str(), load_s(), skip.size());

    strata::core::NativeDense native_dense;
    if (!o.native_dense_gguf.empty()) {
        if (!native_dense.load(o.native_dense_gguf, wt, err, o.native_ple_key)) {
            std::fprintf(stderr, "strata generate: native dense projections: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: %zu native projection matrices, %.2f MiB of weights, in %.1f s\n",
                     native_dense.tensor_count(), (double) native_dense.weight_bytes() / (1024.0 * 1024.0), load_s());
    }

    strata::kernels::gr_set_fp32_activations(o.gr_fp32_activations);
    strata::kernels::gr_set_native_mmvf(o.gr_native_mmvf);
    // Plan v0.3 P3: the fused hyper-connection read rides the native (FP32-activation) contract; the per-stage
    // and dump measurements need the unfused layout of R, so they keep the old kernels.
    strata::core::layer_set_fast_attn(!o.no_fast_attn);
    strata::core::layer_set_publish_kernel(!o.no_publish_kernel);
    strata::core::layer_set_fused_gdn(!o.no_fused_gdn);
    strata::core::layer_set_fast_select(!o.no_fast_select);
    strata::core::layer_set_fused_gr(o.gr_native_mmvf && !o.no_fused_gr && !o.gpu_stages && o.dump_layers.empty() &&
                                     o.dump_halves.empty() && !o.stage_timing);
    strata::core::layer_set_native_bf16(o.native_bf16);
    strata::core::layer_set_native_flash_attn_short(o.native_flash_attn_short);
    strata::kernels::ple_set_native_bf16(o.native_bf16_extra);
    strata::kernels::shared_expert_set_native_bf16(o.native_bf16_extra);
    strata::kernels::native_moe_combine_set_enabled(o.native_moe_combine);
    strata::kernels::native_gdn_set_enabled(o.native_gdn);
    strata::kernels::native_router_set_enabled(o.native_router);
    strata::kernels::native_qsa_set_enabled(o.native_qsa);
    strata::kernels::native_qsa_indexer_set_enabled(o.native_qsa_indexer);
    strata::kernels::native_rope_set_enabled(o.native_rope);
    // The vision path: every rope kernel reads a cell's (t, h, w) from this table (strata/kernels/mrope.hpp).  It is
    // the identity until an image request, and it is set here, before any CUDA graph captures a rope kernel.
    int32_t* d_mrope = nullptr;
    std::vector<int32_t> mrope_host;
    if (o.vision) {
        const int64_t cells = o.max_context + 64;
        mrope_host.resize((size_t) cells * 3);
        for (int64_t c = 0; c < cells; ++c)
            mrope_host[(size_t) c * 3] = mrope_host[(size_t) c * 3 + 1] = mrope_host[(size_t) c * 3 + 2] = (int32_t) c;
        if (!strata::gpu::alloc_device(&d_mrope, mrope_host.size() * sizeof(int32_t)) ||
            !strata::gpu::copy(d_mrope, mrope_host.data(), mrope_host.size() * sizeof(int32_t))) {
            std::fprintf(stderr, "strata generate: cannot allocate the image position table\n");
            return 1;
        }
        strata::kernels::mrope_table_set(d_mrope);
    }
    strata::kernels::ple_set_native_postops(o.native_ple_postops);
    strata::core::ModelGeometry g;   // canonical defaults; the model file overrides the MoE shape below
    int64_t K = 10;
    // the model file's rope keys (llama.cpp's names; the artifact ships none today)
    std::string gguf_rope_type;
    double gguf_rope_base = 0, gguf_rope_factor = 0, gguf_rope_orig_ctx = 0;
    if (!o.native_preset.empty()) {
        // a pruned variant (GSQ-RCO Coder) ships fewer experts than the canonical 512x10; the model file
        // is the authority on its own MoE shape - everything else in the geometry is unchanged
        try {
            strata::GgufFile model_gguf(o.native_preset);
            if (const strata::MetaValue* v = model_gguf.get("qwen4exp.expert_count")) g.n_expert = (int64_t) v->u;
            if (const strata::MetaValue* v = model_gguf.get("qwen4exp.expert_used_count")) K = (int64_t) v->u;
            if (const strata::MetaValue* v = model_gguf.get("qwen4exp.rope.freq_base")) gguf_rope_base = v->num();
            if (const strata::MetaValue* v = model_gguf.get("qwen4exp.rope.scaling.type")) gguf_rope_type = v->s;
            if (const strata::MetaValue* v = model_gguf.get("qwen4exp.rope.scaling.factor")) gguf_rope_factor = v->num();
            if (const strata::MetaValue* v = model_gguf.get("qwen4exp.rope.scaling.original_context_length"))
                gguf_rope_orig_ctx = v->num();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata generate: reading the model's expert shape from %s: %s\n",
                         o.native_preset.c_str(), e.what());
            return 1;
        }
    }
    // THE ROPE CONFIGURATION, before session_init builds the table and the graphs capture the kernels
    {
        using RST = strata::kernels::RopeScalingType;
        if (!o.rope_scaling.empty()) {
            // `none` here is the CLI opting out of the model file's keys
            rope_cfg.type = o.rope_scaling == "linear" ? RST::Linear : o.rope_scaling == "yarn" ? RST::YaRN : RST::None;
        } else if (!gguf_rope_type.empty()) {
            if (gguf_rope_type == "linear") rope_cfg.type = RST::Linear;
            else if (gguf_rope_type == "yarn") rope_cfg.type = RST::YaRN;
            else if (gguf_rope_type != "none") {
                std::fprintf(stderr, "strata generate: %s carries rope.scaling.type '%s' - none, linear or yarn only\n",
                             o.native_preset.c_str(), gguf_rope_type.c_str());
                return 2;
            }
        }
        if (o.rope_scale > 0) rope_cfg.factor = o.rope_scale;
        else if (gguf_rope_factor > 1.0) rope_cfg.factor = gguf_rope_factor;
        if (o.rope_freq_base > 0) rope_cfg.freq_base = o.rope_freq_base;
        else if (gguf_rope_base > 1.0) rope_cfg.freq_base = gguf_rope_base;
        if (o.yarn_orig_ctx > 0) rope_cfg.orig_ctx = o.yarn_orig_ctx;
        else if (gguf_rope_orig_ctx >= 1) rope_cfg.orig_ctx = gguf_rope_orig_ctx;
        rope_cfg.freq_scale_in = o.rope_freq_scale;
        rope_cfg.ext_factor = o.yarn_ext_factor >= 0 ? o.yarn_ext_factor : (rope_cfg.type == RST::YaRN ? 1.0 : 0.0);
        rope_cfg.attn_factor = o.yarn_attn_factor;
        rope_cfg.beta_fast = o.yarn_beta_fast;
        rope_cfg.beta_slow = o.yarn_beta_slow;
        if (rope_cfg.type == RST::None) {
            // the trained rotation exactly: the scaling knobs are inert; only the frequency base survives
            const bool knobs = o.rope_scale > 1.0 || o.rope_freq_scale > 0 || o.yarn_ext_factor > 0 ||
                               o.yarn_attn_factor != 1.0;
            const double base = rope_cfg.freq_base;
            rope_cfg = strata::kernels::RopeScaling{};
            rope_cfg.freq_base = base;
            if (knobs)
                std::fprintf(stderr, "strata generate: note: no rope scaling is active (none), so --rope-scale, "
                                     "--rope-freq-scale and the --yarn-* knobs have no effect\n");
        }
        if (const char* why = strata::kernels::rope_scaling_invalid(rope_cfg)) {
            std::fprintf(stderr, "strata generate: invalid rope scaling configuration: %s (factor %g, freq_scale %g, "
                                 "base %g, original context %g)\n",
                         why, rope_cfg.factor, rope_cfg.freq_scale(), rope_cfg.freq_base, rope_cfg.orig_ctx);
            return 2;
        }
        strata::kernels::rope_scaling_set(rope_cfg);
        if (rope_cfg.type != RST::None) {
            std::fprintf(stderr,
                         "strata generate: rope scaling %s, factor %.6g (freq_scale %.6g, base %.6g, mscale %.6f), "
                         "--max-context %lld against a trained context of %.0f\n",
                         rope_cfg.type == RST::YaRN ? "yarn" : "linear", rope_cfg.factor, rope_cfg.freq_scale(),
                         rope_cfg.freq_base, rope_cfg.mscale(), (long long) o.max_context, rope_cfg.orig_ctx);
            if ((double) o.max_context <= rope_cfg.orig_ctx)
                std::fprintf(stderr, "strata generate: note: the context is within the trained %.0f - no position "
                                     "needs the extension, and the scaling still applies to every angle\n",
                             rope_cfg.orig_ctx);
        }
    }
    // before session_init: every graph captured from here on has the vector's kernels where it applies
    std::string cvec_summary = "0";
    uint64_t cvec_digest = 0;   // a session file is bound to the loaded vector (0: none)
    if (!o.cvec_files.empty()) {
        std::string ce;
        if (!load_control_vectors(o, g, cvec_summary, cvec_digest, ce)) {
            std::fprintf(stderr, "strata generate: control vector: %s\n", ce.c_str());
            return 2;
        }
    }
    if (o.max_context < (int64_t) o.tokens.size() + o.max_new) {
        std::fprintf(stderr, "strata generate: --max-context %lld cannot hold %zu prompt + %lld new tokens\n",
                     (long long) o.max_context, o.tokens.size(), (long long) o.max_new);
        return 2;
    }

    void* sbuf = nullptr;
    // a layer split with explicit split points (upstream #216, the carve): every GPU's sessions hold the state of its
    // own layers only ([0, K) here, [lb, le) on a later stage); --layer-split auto and one GPU hold every layer
    const bool carve = multi_gpu && !split_auto && !split_at.empty();
    const int64_t hi0 = carve ? split_at[0] : -1;
    auto stage_lo = [&](size_t i) { return split_at[i]; };
    auto stage_hi = [&](size_t i) { return i + 1 < split_at.size() ? split_at[i + 1] : g.n_layers; };
    if (!strata::gpu::alloc_device(&sbuf, strata::core::session_bytes(g, o.max_context, K, 0, hi0))) {
        std::fprintf(stderr, "strata generate: session state allocation failed\n");
        return 1;
    }
    strata::core::SessionState ss;
    // All four session calls (`session_capture`, `session_replay`, `session_token` and `session_loop`) were handed
    // `nullptr` (stream 0). `bench/micro/kernel_costs.cu` measures what that costs: EVERY kernel it launches through
    // a wrapper comes back at 28-31 us REGARDLESS OF SIZE, `scale_inplace` on 2,048 floats and `silu_inplace`
    // on 10,240 floats being indistinguishable, which is a fixed per-launch cost and not execution.
    // `bench/micro/graph_node_cost.cu` measures the same kernels on a real stream at 3.63 us ungrapped and
    // 0.805 us inside a graph.  **That is an ~8x penalty on every launch in the engine.**
    strata::gpu::Stream main_stream = nullptr;
    if (!(main_stream = strata::gpu::stream_create())) {
        std::fprintf(stderr, "strata generate: cannot create the main stream\n");
        return 1;
    }
    void* const main_cs = (void*) main_stream;
    if (strata::core::session_init(g, o.max_context, K, sbuf, ss, 0, hi0) == 0) {
        std::fprintf(stderr, "strata generate: session_init failed\n");
        return 1;
    }
    // The session state starts zeroed for every pack (upstream 853aa46, ec64796): a native pack, and a canonical one
    // whose prompt goes through the batched prompt path, start at position 0 without the put_input that zeroed it,
    // and a reused allocation then held overflow/NaN that saturated the layer stack.  --serve zeroes it per request.
    strata::core::session_zero(ss, g, nullptr, main_cs);
    if (!strata::gpu::stream_sync(main_cs)) {
        std::fprintf(stderr, "strata generate: zeroing the session state failed\n");
        return 1;
    }
    if (g.n_qsa_layers() > 0 && ss.qsa_states[ss.qsa_primary()].kv_mode == 1)
        std::fprintf(stderr, "strata generate: KV streaming: %lld of %lld cells per QSA layer in VRAM, the K/V in "
                             "%.2f GiB of pinned RAM\n", (long long) ss.qsa_states[ss.qsa_primary()].n_slots * 4,
                     (long long) o.max_context, (double) strata::core::qsa_kv_host_bytes() / 1073741824.0);

    // ---- **THE HALF-LEVEL DUMP HAS TO BE ARMED BEFORE `session_capture`, AND THE LADDER MUST NOT BE.**  The
    // half copies are issued from inside `block_layer_pre`/`block_layer_post`, so they are only ever enqueued
    // while a graph is being CAPTURED - arming `ss.block.dump` afterwards would produce a file of zeros that
    // reads exactly like a wrong answer.  The ladder is the opposite: `session_loop` enqueues it per token on
    // the replay stream, so it must be armed after capture to stay out of the graph.
    const uint64_t half_stride = (uint64_t) 2 * g.n_embd + (uint64_t) 2 * g.hc +
                                 (uint64_t) g.n_head * g.head_dim +
                                 (uint64_t) 5 * g.n_head_kv * g.head_dim + 8;
    std::FILE* half_dump = nullptr;
    float* half_stage = nullptr;
    if (!o.dump_halves.empty()) {
        half_dump = std::fopen(o.dump_halves.c_str(), "wb");
        if (half_dump == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_halves.c_str());
            return 1;
        }
        const size_t n = (size_t) g.n_layers * (size_t) half_stride;
        if (!strata::gpu::alloc_host((void**) &half_stage, n * sizeof(float))) {
            std::fprintf(stderr, "strata generate: cannot pin the half-dump staging buffer\n");
            return 1;
        }
        ss.block.dump = half_stage;
    }

    strata::core::Doorbell db;
    if (strata::core::doorbell_init(g, K, db) == 0) {
        std::fprintf(stderr, "strata generate: doorbell_init failed\n");
        return 1;
    }
    ss.db = &db;

    // ================================ THE PLE ================================
    //
    // **ITS ABSENCE IS WHY GATE C1 FAILED** (LEDGER L123): layer 1 carries six `blk.1.ple_*` tensors, the whole
    // module was built and parity-tested, and nothing called it.  Everything below is construction - the table
    // is a mapping of the ORIGINAL second GGUF shard, the six weights are already loaded in the arena, and the
    // three buffers are the only allocation.
    strata::kernels::PleTable ple_table;
    std::vector<float> ple_emb_host((size_t) strata::kernels::NG_N_EMBD);
    float* ple_emb_dev = nullptr;
    float* ple_scratch = nullptr;
    if (!o.ple_gguf.empty()) {
        strata::kernels::PleIoOptions pio;
        pio.mode = o.ple_io == "mmap" || o.ple_io == "ram" ? strata::kernels::PleIo::Mmap : strata::kernels::PleIo::Direct;
        pio.lock = o.ple_io == "ram";   // upstream 12fd16c / 9ec069b
        const auto tpl = Clock::now();
        pio.max_inflight = (uint32_t) o.ple_inflight;
        pio.cache_rows = (uint64_t) o.ple_row_cache;
        pio.io_thread = !o.ple_sync_submit;
        // Keep the SSD awake while rows are asked for (PleReader::set_keepalive, upstream e128d6d): some SSDs stall
        // the first reads 50-150 ms after ~250 ms without a command.  STRATA_SSD_KEEPALIVE = ms without a read
        // before one page is read anyway (default 100, 0 = off), STRATA_SSD_KEEPALIVE_WINDOW = seconds after the
        // last row request that this goes on (default 60).
        {
            const char* ka = std::getenv("STRATA_SSD_KEEPALIVE");
            const char* kw = std::getenv("STRATA_SSD_KEEPALIVE_WINDOW");
            pio.keepalive_ms = ka != nullptr && *ka ? std::clamp(std::strtod(ka, nullptr), 0.0, 10000.0) : 100.0;
            pio.keepalive_window_s = kw != nullptr && *kw ? std::clamp(std::strtod(kw, nullptr), 1.0, 86400.0) : 60.0;
            if (pio.mode != strata::kernels::PleIo::Direct || !pio.io_thread) pio.keepalive_ms = 0;
        }
        if (!ple_table.open(o.ple_gguf, err, pio)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        if (pio.lock)
            std::fprintf(stderr, "strata generate: PLE table %s (--ple-io ram) in %.1f s\n",
                         ple_table.locked() ? "locked in RAM" : "loaded (not locked)",
                         std::chrono::duration<double>(Clock::now() - tpl).count());
        std::fprintf(stderr, "strata generate: PLE table format %s\n", ple_table.format());
        if (pio.keepalive_ms > 0)
            std::fprintf(stderr, "strata generate: the SSD is kept awake while rows are read: one page of the table after "
                                 "%.0f ms without a read, until %.0f s after the last request "
                                 "(STRATA_SSD_KEEPALIVE=0 turns it off)\n", pio.keepalive_ms, pio.keepalive_window_s);
        const strata::core::WeightRef* wk = wt.find("blk.1.ple_key.weight");
        const strata::core::WeightRef* wv = wt.find("blk.1.ple_value.weight");
        const strata::core::WeightRef* wnk = wt.find("blk.1.ple_norm_key.weight");
        const strata::core::WeightRef* wnq = wt.find("blk.1.ple_norm_query.weight");
        const strata::core::WeightRef* wnc = wt.find("blk.1.ple_norm_conv.weight");
        const strata::core::WeightRef* wc = wt.find("blk.1.ple_conv1d.weight");
        if (!wk || !wv || !wnk || !wnq || !wnc || !wc) {
            std::fprintf(stderr, "strata generate: the pack has no blk.1.ple_* tensors, so the PLE cannot be "
                                 "wired - and running without it is a DIFFERENT MODEL (LEDGER L123)\n");
            return 1;
        }
        // The conv1d kernel reads F16, so this cast is a claim about the pack's storage.  A checkpoint that keeps
        // the tensor F32 (Q8_0, UD-Q4_K_XL) would hand the kernel the low halves of the f32 words - not an error,
        // a plausible wrong layer-1 routing.  tools/iq_pack.py narrows it (index kind 3); a pack that did not is
        // refused here (upstream bede5ca, #255).  F16 is index kind 5 or 3 (F16InF32) in a native pack and kind 0
        // (verbatim 2-byte F16) in the canonical Q2_0 pack; BF16 (kind 4) has the same size and is not F16.
        const bool conv_f16 = wc->kind == strata::core::WeightKind::F16InF32 ||
                              (wc->kind == strata::core::WeightKind::Verbatim && wc->code_bits == 0);
        if (!conv_f16 || wc->bytes != (uint64_t) wc->elements * 2) {
            std::fprintf(stderr, "strata generate: blk.1.ple_conv1d.weight is not stored as F16 (pack index kind %d, "
                                 "%llu B for %lld values); the PLE conv1d kernel reads F16 - repack with "
                                 "tools/iq_pack.py\n",
                         (int) wc->kind, (unsigned long long) wc->bytes, (long long) wc->elements);
            return 1;
        }
        // `ple_key` is S2 and the loader has already widened its scales to f32, so the two planes are located
        // by the sizes the `WeightRef` records rather than re-derived - the same rule `plane_ptrs` follows.
        if (!wk->quantized()) {
            // plan v0.3 P6: the IQ model files' BF16 key (the pack's extra.bin, raw BF16)
            ss.ple.w.key_bf16 = (const uint16_t*) wk->data;
        } else if (wk->data != nullptr) {
            ss.ple.w.key_codes = (const uint8_t*) wk->data;
            ss.ple.w.key_scales = (const float*) ((const uint8_t*) wk->data + wk->codes_bytes);
        }
        if (o.native_ple_key && wk->quantized()) {
            // the types kernels::ple takes: Q2_0 (42), IQ3_XXS (18), IQ4_XS (23), Q8_0 (8: Unsloth's UD-IQ4_XS; its
            // aligned row layout once loaded)
            if (!wk->native_data || (wk->native_type != 42 && wk->native_type != 18 && wk->native_type != 23 &&
                                     wk->native_type != 8 && wk->native_type != strata::kernels::kNativeQ8Rows) ||
                !wk->native_q8_1) {
                std::fprintf(stderr, "strata generate: native PLE key is absent or incompatible\n");
                return 1;
            }
            ss.ple.w.key_native_data = wk->native_data;
            ss.ple.w.key_native_type = wk->native_type;
            ss.ple.w.key_native_q8_1 = wk->native_q8_1;
        }
        ss.ple.w.value_bf16 = (const uint16_t*) wv->data;
        ss.ple.w.norm_key = (const float*) wnk->data;
        ss.ple.w.norm_query = (const float*) wnq->data;
        ss.ple.w.norm_conv = (const float*) wnc->data;
        ss.ple.w.conv1d_f16 = (const uint16_t*) wc->data;
        ss.ple.consts = strata::kernels::ple_artifact_consts();
        if (o.ple_delay_us > 0) ple_table.set_injected_delay_us(o.ple_delay_us);
        ss.ple.table = &ple_table;
        ss.ple.token = &ss.ple_token;
        ss.ple.prev = ss.ple_prev;
        ss.ple.hist = ss.ple_hist;
        ss.ple.emb_host = ple_emb_host.data();
        if (!strata::gpu::alloc_device((void**) &ple_emb_dev, (size_t) strata::kernels::NG_N_EMBD * 4) ||
            !strata::gpu::alloc_device((void**) &ple_scratch, strata::core::ple_run_scratch_bytes())) {
            std::fprintf(stderr, "strata generate: the PLE buffers failed\n");
            return 1;
        }
        ss.ple.emb_dev = ple_emb_dev;
        ss.ple.scratch = ple_scratch;
        if (!ss.ple.ready()) {
            std::fprintf(stderr, "strata generate: the PLE run is not ready after construction\n");
            return 1;
        }
        std::fprintf(stderr, "strata generate: PLE on, table %llu rows of %s\n",
                     (unsigned long long) ple_table.rows(), o.ple_gguf.c_str());
    } else {
        std::fprintf(stderr,
                     "strata generate: PLE OFF by explicit --no-ple diagnostic request.\n"
                     "  The tokens below are NOT this model's; this is only useful for A/B measurement.\n");
    }

    float* d_parts = nullptr;
    if (!strata::gpu::alloc_device(&d_parts, (size_t) K * g.n_embd * 4) ||
        !strata::gpu::memset(d_parts, 0, (size_t) K * g.n_embd * 4)) {
        std::fprintf(stderr, "strata generate: the parts buffer failed\n");
        return 1;
    }

    strata::core::Verifier::set_commit_async(true);   // see Verifier::set_commit_async
    // ---- layer split across GPUs: each later stage's own copy of the dense weights, its session and (the last) the
    // head, made on its device before the host arena is mapped
    std::vector<std::unique_ptr<GpuStage>> stages;
    for (size_t i = 0; multi_gpu && i < split_devs.size(); ++i) {
        stages.push_back(std::make_unique<GpuStage>());
        GpuStage& st = *stages.back();
        st.dev = split_devs[i];
        const bool last = i + 1 == split_devs.size();
        double free_gib = 0;
        if (!strata::core::RemoteExperts::preflight(st.dev, free_gib, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        const strata::core::OnDevice on(st.dev);
        std::set<std::string> skip_s = skip;
        uint64_t pool_s = pool_bytes;
        if (stage_trim) {
            const int64_t lb = split_at[i], le = i + 1 < split_at.size() ? split_at[i + 1] : g.n_layers;
            skip_s = skip_base;
            add_foreign(lb, le, skip_s);
            strata::core::NativeDense::set_layer_range((int) lb, (int) le);
            if (!strata::core::WeightTable::pool_bytes(o.pack, pool_s, err, &skip_s)) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            std::fprintf(stderr,
                         "strata generate: layer split: CUDA%d loads the dense weights of layers %lld-%lld only\n",
                         st.dev, (long long) lb, (long long) le - 1);
        }
        void* arena_s = nullptr;
        if (!strata::gpu::alloc_device(&arena_s, pool_s) ||
            !st.wt.load(o.pack, arena_s, pool_s, err, skip_s.empty() ? nullptr : &skip_s)) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d weights: %s\n", st.dev,
                         err.empty() ? "the weight arena does not fit" : err.c_str());
            return 1;
        }
        if (!o.native_dense_gguf.empty() && !st.dense.load(o.native_dense_gguf, st.wt, err, o.native_ple_key)) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d native dense projections: %s\n", st.dev,
                         err.c_str());
            return 1;
        }
        void* sbuf_s = nullptr;
        const int64_t slo = carve ? stage_lo(i) : 0, shi = carve ? stage_hi(i) : -1;
        if (!strata::gpu::alloc_device(&sbuf_s, strata::core::session_bytes(g, o.max_context, K, slo, shi)) ||
            strata::core::session_init(g, o.max_context, K, sbuf_s, st.ss, slo, shi) == 0 ||
            !(st.stream = strata::gpu::stream_create()) ||
            !(st.adapt_stream = strata::gpu::stream_create()) ||
            !strata::gpu::event_create(&st.adapt_ev)) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d: the session state failed\n", st.dev);
            return 1;
        }
        const strata::core::WeightRef* wo_s = st.wt.find("output.weight");
        if (wo_s == nullptr ||
            (last && !o.native_head_gguf.empty() && !st.head.load(o.native_head_gguf, g.n_embd, wo_s->ne1, err))) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d head: %s%s\n", st.dev,
                         wo_s == nullptr ? "output.weight is missing" : err.c_str(),
                         wo_s == nullptr ? "" : vram_free_note().c_str());
            return 1;
        }
        // a control vector (the speed projection): its tables on this device too - the stage's layers apply it here
        if (!strata::kernels::cvec_replicate(err)) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d: %s\n", st.dev, err.c_str());
            return 1;
        }
        // --vision: this device's image-position table (the identity until a picture request), read by every rope
        // kernel its stage runs - set before any of its graphs is captured
        if (o.vision) {
            if (!strata::gpu::alloc_device(&st.mrope, mrope_host.size() * sizeof(int32_t)) ||
                !strata::gpu::copy(st.mrope, mrope_host.data(), mrope_host.size() * sizeof(int32_t))) {
                std::fprintf(stderr, "strata generate: layer split, CUDA%d: the image position table failed\n", st.dev);
                return 1;
            }
            strata::kernels::mrope_table_set(st.mrope);
        }
        // its own PCIe share of the missed experts (the same rule as CUDA0's above: its link is probed)
        st.pcie_frac = o.pcie_frac;
        if (!pcie_given && native_pack) {
            std::string bursts;
            const double bw = probe_pcie_h2d_gbps(&bursts);
            if (bw > 0.0) st.pcie_frac = pcie_frac_for_gbps(bw, 0.55);
            std::fprintf(stderr, "strata generate: layer split: CUDA%d PCIe probe %.1f GB/s (best of %s) -> pcie_frac "
                                 "%.2f\n", st.dev, bw, bursts.c_str(), st.pcie_frac);
        }
        size_t fb = 0, tb = 0;
        strata::gpu::mem_info(&fb, &tb);
        std::fprintf(stderr, "strata generate: layer split: CUDA%d holds its weights, session%s; %.2f GiB free\n",
                     st.dev, last ? " and head" : "", (double) fb / 1073741824.0);
    }
    GpuStage* const last_st = stages.empty() ? nullptr : stages.back().get();

    // --batch (upstream PR #559, #465): o.batch more sessions carved like the main one, one per slot of the batch
    // windows.  Before the expert cache is sized, so `--expert-cache auto` leaves them room.  A count the engine
    // cannot run is a warning and fewer slots or none, never a refusal.
    std::vector<std::unique_ptr<strata::core::SessionState>> bslot_ss;
    std::vector<void*> bslot_buf;
    if (o.batch != 0) {
        const char* why = !o.serve ? "it needs --serve"
                        : o.batch < 2 ? "it needs at least 2 slots"
                        : nullptr;
        if (o.batch > strata::kernels::kVerifyMaxT) {
            std::fprintf(stderr, "strata generate: --batch %d: a window holds %d rows, so %d slots\n", o.batch,
                         strata::kernels::kVerifyMaxT, strata::kernels::kVerifyMaxT);
            o.batch = strata::kernels::kVerifyMaxT;
        }
        if (why != nullptr) {
            std::fprintf(stderr, "strata generate: --batch is off: %s\n", why);
            o.batch = 0;
        }
        const uint64_t bytes = strata::core::session_bytes(g, o.max_context, K, 0, hi0);
        for (int b = 0; b < o.batch; ++b) {
            auto u = std::make_unique<strata::core::SessionState>();
            void* buf = nullptr;
            if (!strata::gpu::alloc_device(&buf, bytes) ||
                strata::core::session_init(g, o.max_context, K, buf, *u, 0, hi0) == 0) {
                if (buf) strata::gpu::free(buf);
                std::fprintf(stderr, "strata generate: --batch: slot %d's session does not fit (%.2f GiB each)\n", b,
                             (double) bytes / 1073741824.0);
                break;
            }
            strata::core::session_zero(*u, g, nullptr, main_cs);
            bslot_ss.push_back(std::move(u));
            bslot_buf.push_back(buf);
        }
        // a layer split: every later GPU keeps the slots' state of its own layers in sessions of its own (one GPU in
        // two stages shares them: --split-device 0); as many slots as every GPU holds
        for (auto& stp : stages) {
            GpuStage& st = *stp;
            const strata::core::OnDevice on(st.dev);
            // carved like the stage's own session (its layer range)
            const uint64_t sbytes = strata::core::session_bytes(g, o.max_context, K, st.ss.layer_lo, st.ss.layer_hi);
            while (st.bslots.size() < bslot_ss.size()) {
                auto u = std::make_unique<strata::core::SessionState>();
                void* buf = nullptr;
                if (!strata::gpu::alloc_device(&buf, sbytes) ||
                    strata::core::session_init(g, o.max_context, K, buf, *u, st.ss.layer_lo, st.ss.layer_hi) == 0) {
                    if (buf) strata::gpu::free(buf);
                    std::fprintf(stderr, "strata generate: --batch: CUDA%d holds %zu slot sessions\n", st.dev,
                                 st.bslots.size());
                    break;
                }
                strata::core::session_zero(*u, g, nullptr, st.stream);
                st.bslots.push_back(std::move(u));
                st.bslot_buf.push_back(buf);
            }
            while (bslot_ss.size() > st.bslots.size()) {
                strata::gpu::free(bslot_buf.back());
                bslot_buf.pop_back();
                bslot_ss.pop_back();
            }
        }
        for (auto& stp : stages)
            while (stp->bslots.size() > bslot_ss.size()) {
                const strata::core::OnDevice on(stp->dev);
                strata::gpu::free(stp->bslot_buf.back());
                stp->bslot_buf.pop_back();
                stp->bslots.pop_back();
            }
        if (o.batch > 0 && bslot_ss.size() < 2) {
            std::fprintf(stderr, "strata generate: --batch is off: not two slot sessions fit\n");
            for (void* b : bslot_buf) strata::gpu::free(b);
            bslot_ss.clear();
            bslot_buf.clear();
        }
        o.batch = (int) bslot_ss.size();
        if (o.batch > 0) {
            strata::gpu::stream_sync(main_cs);
            size_t fb = 0, tb = 0;
            strata::gpu::mem_info(&fb, &tb);
            std::fprintf(stderr, "strata generate: --batch: %d slot sessions (%.2f GiB each); %.2f GiB free\n", o.batch,
                         (double) bytes / 1073741824.0, (double) fb / 1073741824.0);
        }
    }

    // Allocate the MTP weights before registering the large host arena with both CUDA contexts.
    strata::core::MtpDrafter mtp;
    if (!o.mtp.empty()) {
        if (o.spec < 2) {
            std::fprintf(stderr, "strata generate: --mtp is ignored without --spec T (T >= 2)\n");
            o.mtp.clear();
        }
        if (!o.mtp.empty()) mtp.set_prompt_len((int64_t) o.tokens.size());
        mtp.set_hnorm_per_stream(o.mtp_hnorm_stream);   // --mtp-hnorm stream (opt-in), before any capture
        // the draft layer is the canonical model's MTP head (512 experts) even when the target is pruned,
        // so it always sees the canonical geometry; `static` because MtpDrafter keeps a reference
        static const strata::core::ModelGeometry draft_geometry{};
        // with a layer split across GPUs the drafter reads the last stage's residual: it lives on that device
        const strata::core::OnDevice on_mtp(last_st ? last_st->dev : -1);
        if (!o.mtp.empty() && !mtp.load(o.mtp, draft_geometry, last_st ? last_st->ss : ss, o.spec, err, o.mtp_window)) { std::fprintf(stderr, "strata generate: %s%s\n", err.c_str(), vram_free_note().c_str()); return 1; }
    }
    // THE HEAD BEFORE THE CACHE, AND BEFORE THE ARENA.  The expert cache takes what is free minus the reserve, so
    // everything allocated after it comes out of the reserve.  The native head (~0.5 GB with IQ3_S) was loaded after
    // it and ate most of the 700 MiB: 128K IQ3_S ended with 30 MiB free, the driver paged, and a request stalled for
    // good at its first verify window.  Loaded first, the cache is sized around it.  #620: and before the expert
    // arena registers tens of GiB of host pages, like the drafter above (on Windows the driver then refused the
    // head's allocation with GiBs free).
    const strata::core::WeightRef* wo = wt.find("output.weight");
    if (wo == nullptr) { std::fprintf(stderr, "strata generate: output.weight is missing\n"); return 1; }
    const int64_t n_vocab = wo->ne1;
    strata::core::NativeHead native_head;
    if (!o.native_head_gguf.empty() && !multi_gpu) {   // a layer split's head is on its last stage
        const auto head_t0 = std::chrono::steady_clock::now();
        if (!native_head.load(o.native_head_gguf, g.n_embd, n_vocab, err)) {
            std::fprintf(stderr, "strata generate: %s%s\n", err.c_str(), vram_free_note().c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: experimental native Q5_K head, %llu bytes, in %.1f s\n",
                     (unsigned long long) native_head.weight_bytes(),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - head_t0).count());
    }
    std::vector<float> logits((size_t) n_vocab);
    float* d_logits = nullptr;
    if (!strata::gpu::alloc_device(&d_logits, (size_t) n_vocab * 4)) {
        std::fprintf(stderr, "strata generate: the logits buffer failed%s\n", vram_free_note().c_str());
        return 1;
    }
    // Create the additional contexts after MTP has secured CUDA0 memory, but
    // before the host arena maps its expert pages into their address spaces.
    for (int r = 1; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0) {
        double free_gib = 0;
        if (!strata::core::RemoteExperts::preflight(remote_dev[r], free_gib, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: CUDA%d context ready, %.2f GiB free before expert arena registration\n",
                     remote_dev[r], free_gib);
    }

    // ---- the CPU expert pool
    //
    // R2.1: the experts are loaded into a RESIDENT ARENA by default.  The mmap path is kept behind
    // `--mmap-experts` because it is the A/B arm, not because it is competitive.
    //
    // The reasoning is the review's C1 and it is now measured on both sides.  `FileExpertSource` maps the 34 GB
    // file, and mapped file pages are the first thing the OS reclaims; the engine's rate then depends on whether
    // the standby list happens to hold `experts.bin`, which is why two consecutive runs of the SAME BINARY with
    // the SAME FLAGS measured 71.97 and 34.78 ms/token in the pool (7.54 vs 12.18 tok/s).  The arena is
    // anonymous memory the engine owns, and the pool runs at 19.41 ms/token - 1.79x better than the warm mmap
    // and 3.7x better than the cold one.
    //
    // IT IS NOT PINNED, and that is reported rather than hidden: `cudaHostRegister` on 31.64 GiB fails with
    // "out of memory" (you cannot pin 34 of 63 GB) and the arena falls back to 4 KB anonymous pages.  That is
    // fine for the CPU pool - which is all that exists today - and NOT fine for Phase 3, whose cache fills and
    // CPU/PCIe miss split need the GPU to DMA out of this arena.  Read `note()` when that lands.
    //
    // The earlier "the arena does not fit" conclusion was WRONG and is worth recording: the failure was a stale
    // CUDA error left set by the failed `cudaHostRegister` and read later by `gr_read`'s launch check.  See the
    // note in `pinned.cu`.
    strata::core::FileExpertSource src;
    strata::core::ArenaExpertSource arena_src;
    strata::core::ExpertSource* srcp = nullptr;
    std::vector<uint8_t> fill_tmp;   // blob_to_copy's buffer for a transient source
    if (o.mmap_experts) {
        // FileExpertSource maps the pack's experts.bin: a canonical pack has it; a native (IQ) pack has it when
        // built with `tools/iq_pack.py --experts-bin` (per-layer blob sizes, upstream de159b5).  The low-RAM mode:
        // the experts come from the file through the OS cache instead of a copy in RAM, for a PC whose GPU holds
        // most of them but whose RAM cannot hold them all.
        // Without experts.bin a native pack's experts are read from the model's GGUF files in place (--native;
        // upstream 3889344): each blob assembled from its gate, up and down slices when it is needed.
        if (native_pack && !std::ifstream(o.pack + "/experts.bin", std::ios::binary)) {
            if (o.native_preset.empty()) {
                std::fprintf(stderr, "strata generate: --mmap-experts on the native pack %s without experts.bin needs "
                                     "--native (the experts are read from the GGUF files)\n", o.pack.c_str());
                return 2;
            }
            src.set_gguf(o.native_preset);
        }
        if (!src.open(o.pack, g.n_layers, g.n_expert, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: experts via mmap (--mmap-experts: %s, through the OS file cache)\n",
                     src.gguf_mode() ? "the GGUF files in place" : "experts.bin");
        if (const char* v = std::getenv("STRATA_FETCH_THREADS")) {
            const long threads = std::strtol(v, nullptr, 10);
            if (threads > 0) src.set_fetch_threads((int) std::min<long>(threads, std::numeric_limits<int>::max()));
        }
        srcp = &src;
    } else {
        arena_src.set_gguf(o.native_preset);   // plan v0.3 P6: a native pack may take its experts from shard 1
        if (!arena_src.open(o.pack, g.n_layers, g.n_expert, /*threads=*/6, err, o.shared_expert_arena)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        if (!arena_src.ram_warning().empty())   // #633: said before the load's numbers, which it explains
            std::fprintf(stderr, "strata generate: WARNING: %s\n", arena_src.ram_warning().c_str());
        std::fprintf(stderr, "strata generate: expert arena: %s\n", arena_src.note().c_str());
        std::fprintf(stderr, "strata generate: loaded %.2f GiB at %.2f GiB/s\n",
                     (double) strata::kernels::cpu::expert_layout().total / (1024.0 * 1024 * 1024),
                     arena_src.load_gib_per_second());
        srcp = &arena_src;
    }
    // --batch-cpu-split (EXPERIMENTAL): the CPU workers in two pools on separate cores, N0 for stage 0 and N1 for
    // stage 1 (a layer split over two GPUs with --batch-groups; anything else keeps the one pool)
    const bool cpu_split = o.batch_cpu_split[0] > 0 && multi_gpu && split_devs.size() == 1 && o.batch_groups > 1 &&
                           o.batch > 0;
    if (o.batch_cpu_split[0] > 0 && !cpu_split)
        std::fprintf(stderr, "strata generate: --batch-cpu-split is off: it needs --batch with --batch-groups over a "
                             "layer split on two GPUs\n");
    const int pipe_n0 = cpu_split ? o.batch_cpu_split[0] : 0, pipe_n1 = cpu_split ? o.batch_cpu_split[1] : 0;
    strata::kernels::cpu::ExpertPool pool(pipe_n0 > 0 ? pipe_n0 : o.pool_workers, /*pin=*/true,
                                          /*host_works=*/!o.no_host_worker, /*first_core=*/0, o.pool_tasks);
    std::fprintf(stderr, "strata generate: CPU pool tasks/phase: %d%s, participating threads: %d\n",
                 o.pool_tasks ? o.pool_tasks : 3 * (pool.workers() + (pool.host_works() ? 1 : 0)),
                 o.pool_tasks ? " (capped by rows)" : " (automatic)",
                 pool.workers() + (pool.host_works() ? 1 : 0));
    std::unique_ptr<strata::kernels::cpu::ExpertPool> pool1;
    if (pipe_n1 > 0) {
        pool1 = std::make_unique<strata::kernels::cpu::ExpertPool>(pipe_n1, true, !o.no_host_worker, pipe_n0,
                                                                   o.pool_tasks);
        std::fprintf(stderr, "strata generate: --batch-cpu-split (experimental): CPU experts in two pools, %d + %d "
                             "workers\n", pipe_n0, pipe_n1);
    }
    // ---- R4's slot storage.  Allocated AFTER the weights and the session, so `cudaMemGetInfo` inside `open`
    // sees the memory this process actually has left rather than the card's idle figure - and refuses with both
    // numbers if the slots do not fit, instead of handing back a cache smaller than it was asked for.
    mem_mark("the weights, the session and the drafter");
    strata::core::ExpertCache xcache;
    // #533: --vram-elastic: the cache in physical segments (the VRAM command resizes it between requests).  Serve
    // mode, one GPU, no helper caches: anything else keeps the one allocation, said once.
    if (o.vram_elastic) {
        const char* why = !o.serve ? "it works between requests of --serve"
                        : multi_gpu ? "a layer split has a cache per GPU"
                        : std::any_of(o.expert_cache_remote.begin(), o.expert_cache_remote.end(),
                                      [](int n) { return n > 0; }) ? "the helper caches on other GPUs"
                        : o.resident_cpu_experts || o.resident_gib != 0 ? "the resident low-RAM mode"
                        : o.vram_segment_mib < 64 ? "--vram-segment-mib is below 64"
                        : !strata::gpu::vmem_supported() ? "this GPU's runtime has no virtual memory" : nullptr;
        if (why != nullptr) {
            std::fprintf(stderr, "strata generate: --vram-elastic is off: %s\n", why);
            o.vram_elastic = false;
        } else {
            xcache.set_segment_bytes(o.vram_segment_mib << 20);
        }
    }
    std::vector<std::pair<int32_t, int32_t>> profile;
    if (!o.expert_profile.empty()) {
        int64_t pslots = 0;
        if (!strata::core::read_expert_profile(o.expert_profile, g.n_layers, g.n_expert, profile, pslots, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        // An explicit number truncates the ranked list ("what would 2,000 slots give" without rebuilding the
        // file).  `--expert-cache 0` used to take the count the profile was built for; the profile now ranks
        // every pair (issue #46: a card that holds more than the old 8,000 used to stop there), so it means auto.
        if (o.expert_cache == 0) o.expert_cache = -1;
        std::fprintf(stderr, "strata generate: profile %s: %zu ranked pairs, built for %lld slots\n",
                     o.expert_profile.c_str(), profile.size(), (long long) pslots);
    }
    // #477: the whole ranking as loaded, the prior of --expert-profile-save's order (a layer split keeps only
    // CUDA0's pairs in `profile` below).  Empty without --expert-profile-save.
    std::vector<std::pair<int32_t, int32_t>> profile_loaded;
    std::vector<std::pair<int32_t, int32_t>> profile_all;   // a layer split: the whole ranking
    if (!o.expert_profile_save.empty()) profile_loaded = profile;
    // ---- layer split across GPUs: "auto" places the split points by a cost model of one decode window, measured on
    // the 5080 + 3090 rig (bench/results/2026-09-29-layer-split):
    //   - every layer costs its GPU a time inversely proportional to SMs x clock (0.33 ms on an RTX 5080, 0.50 on a
    //     3090: the per-layer round trip and kernels, not the bytes - both cards have ~950 GB/s);
    //   - an expert no cache holds costs ~190 ms per unit of routed mass: the CPU pool in decode and the PCIe stream
    //     in prompts (fitted: the sweep's best K, 26-28, is where one more layer on the faster card stops paying
    //     for the ~0.1% of the mass it pushes out of its cache);
    //   - which experts a cache holds: its layers' profiled pairs, hottest first, until its free VRAM (less the
    //     reserve, the prompt path's buffers and, on a later GPU, 1 GiB for its windows and the drafter) is used;
    //     the routed mass of rank r is taken as (r+1)^-1.2 (fits the sweep's hit rates: K=24/26/28 predicted
    //     99.53/99.34/99.15%, measured 99.5/99.4/99.0%).
    // Up to 3 GPUs every placement is tried; beyond, the layers are shared in proportion to speed.
    // STRATA_SPLIT_MISS_MS tunes the miss cost (a slower CPU: higher).
    // A stage's prompt path borrows its buffers from the stage's expert cache, as one GPU's does; without a profile or
    // with --no-prefill-borrow it has buffers of its own: its chunk's buffers and the streamed ring, as
    // Prefill::bytes_needed counts them on that GPU.  (Upstream's 160 MiB + 680 KiB a token left the ring out: ~1 GiB
    // of IQ3_S blobs at 384 slots, and on the B70 with --layer-split 34 the cache left 12 MiB free and the first
    // verify window failed to launch.)
    auto stage_room = [&](int dev, bool later, const strata::core::SessionState& s) -> int64_t {
        const strata::core::OnDevice on(dev);
        size_t fb = 0, tb = 0;
        if (const bool e = strata::gpu::mem_info(&fb, &tb); !e)
            std::fprintf(stderr, "strata generate: layer split: CUDA%d free memory: %s\n", dev < 0 ? 0 : dev,
                         strata::gpu::last_error());
        const bool own = o.no_prefill_borrow || o.expert_profile.empty();
        const int64_t pf = o.prefill_chunk > 0 && own
                               ? (int64_t) strata::prefill::Prefill::bytes_needed(g, s, o.prefill_chunk) : 0;
        const int64_t base_reserve =
            later && o.vram_reserve_later_mib >= 0 ? o.vram_reserve_later_mib : o.vram_reserve_mib;
        const int64_t reserve = ((base_reserve + (later ? 1024 : 0)) << 20) + pf;
        return std::max<int64_t>((int64_t) fb - reserve, 0);
    };
    for (size_t i = 0; i < split_at.size(); ++i)
        if (split_at[i] >= g.n_layers) {
            std::fprintf(stderr, "strata generate: --layer-split: layer %lld is past the last (%lld)\n",
                         (long long) split_at[i], (long long) (g.n_layers - 1));
            return 2;
        }
    // the stage that runs a layer (0: CUDA0's)
    auto stage_of = [&](int64_t l) -> int {
        int st = 0;
        while (st < (int) split_at.size() && l >= split_at[(size_t) st]) ++st;
        return st;
    };
    if (multi_gpu) {
        std::vector<std::pair<int32_t, int32_t>> mine;
        for (const auto& pr : profile) {
            const int st = stage_of(pr.first);
            (st == 0 ? mine : stages[(size_t) st - 1]->profile).push_back(pr);
        }
        profile_all = profile;   // the RAM tier ranks every stage's layers (pin_cache_complement)
        profile.swap(mine);
        for (size_t i = 0; i < stages.size(); ++i) {
            stages[i]->lb = split_at[i];
            stages[i]->le = i + 1 < stages.size() ? split_at[i + 1] : g.n_layers;
        }
    }
    // THE HEAD BEFORE THE CACHE: the native head and the logits are allocated above, before the expert arena (#620)
    const bool auto_cache = o.expert_cache < 0;
    bool reserve_adapted = false;   // #496: the auto sizing lowered the reserve so a small card's cache fits
    if (o.expert_cache < 0) {
        size_t free_b = 0, total_b = 0;
        strata::gpu::mem_info(&free_b, &total_b);
        // The batched prompt path's chunk buffers are allocated later, so reserve room for them here.
        // (with borrowing - the default with a profile - the prompt path lends cache slots instead)
        const bool borrow = !o.no_prefill_borrow && !o.expert_profile.empty();
        // a layer split's first stage: its prompt path's buffers and ring, counted (see stage_room)
        const int64_t prefill_mib =
            (o.prefill_chunk > 0 && !borrow)
                ? (multi_gpu ? (int64_t) (strata::prefill::Prefill::bytes_needed(g, ss, o.prefill_chunk) >> 20) + 1
                             : 160 + (o.prefill_chunk * 680) / 1024)
                : 0;
        // the draft layer's head and logits are allocated when it binds, after this: the CJK draft subset makes them
        // ~110-180 MiB larger, and out of the reserve they left 16 GB cards below the stall line (upstream b981f63)
        const int64_t mtp_bind = (!o.mtp.empty() && native_head.loaded())
                                     ? (int64_t) mtp.bind_bytes(native_head.row_bytes(), n_vocab) : 0;
        const int64_t reserve = (((int64_t) o.vram_reserve_mib + prefill_mib) << 20) + mtp_bind;
        const int64_t blob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
        // Not rounded: free VRAM moves the cache by a slot or two between starts, and with it the last bits of the
        // output (bench/results/2026-10-02-new-machine), but rounding to 64 slots cost the RTX 4070 3-4% of its decode.
        // Runs that must give the same tokens fix the size with --expert-cache N.
        int64_t slots = ((int64_t) free_b - reserve) / blob;
        if (!profile.empty()) slots = std::min<int64_t>(slots, (int64_t) profile.size());
        o.expert_cache = (int) std::max<int64_t>(slots, 0);
        std::fprintf(stderr, "strata generate: expert cache auto: %.2f GiB free, %d MiB reserved (+%lld MiB for the "
                             "draft head) -> %d slots\n",
                     (double) free_b / 1073741824.0, o.vram_reserve_mib, (long long) (mtp_bind >> 20), o.expert_cache);
        // #496 (upstream e1ee248, 4731a9b): the verify window cannot start without a cache, and a cache too small to
        // lend the prompt path a 256-token chunk's buffers (plus the 128 slots a loan leaves; one slot without
        // --prefill) makes it allocate its own on top - more than the reserve.  When the default reserve leaves less
        // than that (a small card), the reserve shrinks to what leaves exactly that cache, down to kSmallReserveMib:
        // what is allocated after the cache - the prompt path's own part, the verify buffers, the draft head - comes
        // out of the reserve, so the cache gets no more than it needs, and the serve check says so plainly when it
        // ends LOW (`reserve_adapted`).  A reserve given on the command line is kept.  No slot at all: the start
        // stops, saying what is short and what makes room.  A card the default reserve leaves that much is sized as
        // before.
        constexpr int kSmallReserveMib = 300;
        const int64_t min_slots = (o.prefill_chunk > 0 && borrow)
            ? ((int64_t) strata::prefill::Prefill::bytes_needed(g, ss, 256) + blob - 1) / blob + 128 : 1;
        if (o.expert_cache < min_slots && !o.vram_reserve_given && o.vram_reserve_mib > kSmallReserveMib) {
            // the largest reserve (in MiB) that still leaves min_slots
            const int64_t fit_mib = ((int64_t) free_b - mtp_bind - min_slots * blob) / (1 << 20) - prefill_mib;
            if (fit_mib >= kSmallReserveMib) {
                const int r = (int) std::min<int64_t>(fit_mib, o.vram_reserve_mib);
                int64_t s2 = ((int64_t) free_b - ((((int64_t) r + prefill_mib) << 20) + mtp_bind)) / blob;
                // not below the slots a working cache needs (s2 >= min_slots: r leaves them)
                s2 = std::max<int64_t>(min_slots, s2);
                if (!profile.empty()) s2 = std::min<int64_t>(s2, (int64_t) profile.size());
                std::fprintf(stderr, "strata generate: expert cache auto: the %d MiB reserve leaves too few slots on "
                                     "this card (a working cache needs %lld): a %d MiB reserve instead -> %lld slots\n",
                             o.vram_reserve_mib, (long long) min_slots, r, (long long) s2);
                o.vram_reserve_mib = r;
                o.expert_cache = (int) s2;
                reserve_adapted = true;
            }
        }
        if (o.expert_cache == 0) {
            // what is short, and what makes room: the numbers a small card picks from
            const int64_t at_reserve = o.vram_reserve_given ? o.vram_reserve_mib
                                                            : std::min(o.vram_reserve_mib, kSmallReserveMib);
            const int64_t need_b = (((int64_t) at_reserve + prefill_mib) << 20) + mtp_bind + min_slots * blob;
            const int64_t short_mib = std::max<int64_t>(1, (need_b - (int64_t) free_b + (1 << 20) - 1) >> 20);
            const int64_t session_mib = (int64_t) (strata::core::session_bytes(g, o.max_context, K) >> 20);
            const std::string reserve_tip =
                o.vram_reserve_given && o.vram_reserve_mib > kSmallReserveMib
                    ? ", a smaller --vram-reserve-mib (" + std::to_string(o.vram_reserve_mib) + " now; " +
                          std::to_string(kSmallReserveMib) + " is enough on a small card)"
                    : std::string();
            std::fprintf(stderr, "strata generate: no VRAM is left for the expert cache: it needs at least %lld slots "
                                 "(%lld MiB), about %lld MiB more than this card has free. To make room: a smaller "
                                 "--max-context (the session, mostly its KV cache, takes %lld MiB at %lld tokens), "
                                 "--kv q4_0, the English draft subset (setup --draft-vocab en; the draft head takes "
                                 "%lld MiB now)%s, images on the CPU, or close other programs that use the GPU\n",
                         (long long) min_slots, (long long) ((min_slots * blob) >> 20), (long long) short_mib,
                         (long long) session_mib, (long long) o.max_context, (long long) (mtp_bind >> 20),
                         reserve_tip.c_str());
        }
    } else if (multi_gpu && o.expert_cache > 0 && o.no_prefill_borrow) {
        // a layer split's prompt path with buffers of its own (--no-prefill-borrow): an explicit cache size leaves
        // room for them and the reserve, or the first prompt fails with "device buffers ... do not fit"
        size_t free_b = 0, total_b = 0;
        strata::gpu::mem_info(&free_b, &total_b);
        const int64_t prefill_mib = o.prefill_chunk > 0 ? 160 + (o.prefill_chunk * 680) / 1024 : 0;
        const int64_t reserve = ((int64_t) o.vram_reserve_mib + prefill_mib) << 20;
        const int64_t fit = std::max<int64_t>(((int64_t) free_b - reserve) / (int64_t) strata::kernels::cpu::expert_layout().max_blob, 0);
        if (o.expert_cache > fit) {
            std::fprintf(stderr, "strata generate: layer split: --expert-cache %d leaves no room for the prompt path's "
                                 "buffers (%lld MiB) on CUDA0: %lld slots\n", o.expert_cache, (long long) prefill_mib,
                         (long long) fit);
            o.expert_cache = (int) fit;
        }
    }
    // plan v0.3 P6: a native pack's blobs differ per layer, so with a profile its slots are sized per pair: the
    // same VRAM holds ~30% more IQ3_XXS experts than slots of the largest blob would
    std::vector<int64_t> sized_slots;
    if (native_pack && o.expert_cache > 0 && !profile.empty()) {
        size_t free_b = 0, total_b = 0;
        strata::gpu::mem_info(&free_b, &total_b);
        const auto& lay = strata::kernels::cpu::expert_layout();
        const uint64_t budget = (uint64_t) o.expert_cache * lay.max_blob;   // what the uniform sizing granted
        uint64_t used = 0;
        size_t free_room = free_b > ((size_t) o.vram_reserve_mib << 20) ? free_b - ((size_t) o.vram_reserve_mib << 20) : 0;
        uint64_t cap = std::min<uint64_t>(budget, (uint64_t) free_room);
        for (const auto& pr : profile) {
            const uint64_t b = (lay.blob_bytes(pr.first) + 255) / 256 * 256;
            if (used + b > cap) break;
            used += b;
            sized_slots.push_back((int64_t) lay.blob_bytes(pr.first));
        }
        // upstream #831: the sized-slots loop walks the profile's pairs, so the profile's length is the ceiling; an
        // explicit --expert-cache above it used to be cut with no word.  A warning only (the allocation is the same).
        if (!auto_cache && o.expert_cache > 0 && sized_slots.size() == profile.size() && (size_t) o.expert_cache > profile.size())
            std::fprintf(stderr, "strata generate: WARNING: --expert-cache %d is more than the %zu pairs of the profile %s: "
                                 "%zu slots (the profile is the ceiling on a native pack; a profile that ranks every "
                                 "(layer, expert) pair lifts it - tools/make_profile.py)\n", o.expert_cache,
                         profile.size(), o.expert_profile.c_str(), profile.size());
        o.expert_cache = (int) sized_slots.size();
    }
    if (o.expert_cache > 0) {
        // keep the first `keep_bytes` of the cache (the profile's hottest experts first); false when nothing is left
        auto shrink_to = [&](int64_t keep_bytes) -> bool {
            if (keep_bytes <= 0) { o.expert_cache = 0; sized_slots.clear(); return false; }
            if (!sized_slots.empty()) {
                int64_t used = 0;
                size_t keep = 0;
                while (keep < sized_slots.size() && used + (sized_slots[keep] + 255) / 256 * 256 <= keep_bytes)
                    used += (sized_slots[keep++] + 255) / 256 * 256;
                sized_slots.resize(keep);
                o.expert_cache = (int) keep;
            } else {
                o.expert_cache = (int) (keep_bytes / (int64_t) strata::kernels::cpu::expert_layout().max_blob);
            }
            if (o.expert_cache <= 0) { o.expert_cache = 0; sized_slots.clear(); return false; }
            return true;
        };
        auto cache_bytes = [&]() -> int64_t {
            if (sized_slots.empty()) return (int64_t) o.expert_cache * (int64_t) strata::kernels::cpu::expert_layout().max_blob;
            int64_t b = 0;
            for (const int64_t s : sized_slots) b += (s + 255) / 256 * 256;
            return b;
        };
        // With `--expert-cache auto` the reserve must still be free once the slots are written. Zero the slots,
        // check free memory again, and reopen a smaller cache if the reserve is short.
        // STRATA_TEST_CACHE_FAIL=N: the first N opens fail as an out-of-commit cudaMalloc does (tests the retry)
        int fake_fails = std::getenv("STRATA_TEST_CACHE_FAIL") ? std::atoi(std::getenv("STRATA_TEST_CACHE_FAIL")) : 0;
        int failed = 0;
        for (int attempt = 0;; ++attempt) {
            bool ok = false;
            if (fake_fails > 0) {
                --fake_fails;
                err = "ExpertCache: device allocation failed: out of memory (STRATA_TEST_CACHE_FAIL)";
            } else {
                ok = sized_slots.empty()
                    ? xcache.open(o.expert_cache, g.n_layers, g.n_expert, (int64_t) strata::kernels::cpu::expert_layout().max_blob, err)
                    : xcache.open_sized(sized_slots, g.n_layers, g.n_expert, err);
            }
            if (!ok) {
                // An auto cache retries a smaller size if the allocation fails.
                if (auto_cache && failed < 8 && shrink_to(cache_bytes() / 4 * 3)) {
                    ++failed;
                    std::fprintf(stderr, "strata generate: %s; trying a smaller expert cache: %d slots\n", err.c_str(),
                                 o.expert_cache);
                    continue;
                }
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            if (!auto_cache || attempt - failed >= 6) break;
            strata::gpu::memset(xcache.device_slot(0), 0, (size_t) xcache.bytes());
            strata::gpu::device_sync();
            size_t free_b = 0, total_b = 0;
            strata::gpu::mem_info(&free_b, &total_b);
            const int64_t want = (int64_t) o.vram_reserve_mib << 20;
            if ((int64_t) free_b >= want - (64ll << 20)) break;
            // short by (want - free); a figure of 0 only says "at least", so then give back a quarter as well
            int64_t give = want - (int64_t) free_b + (64ll << 20);
            if (free_b < ((size_t) 16 << 20)) give = std::max<int64_t>(give, xcache.bytes() / 4);
            const int64_t keep_bytes = xcache.bytes() - give;
            std::fprintf(stderr, "strata generate: only %lld MiB free once the slots are written (reserve %d MiB); "
                                 "shrinking the expert cache\n", (long long) (free_b >> 20), o.vram_reserve_mib);
            xcache.close();
            if (!shrink_to(keep_bytes)) break;
        }
        if (failed > 0 && o.expert_cache > 0)
            std::fprintf(stderr, "strata generate: expert cache: %d slots (%.2f GiB) after %d smaller tries - a bigger "
                                 "page file lets it use more of the free VRAM\n",
                         o.expert_cache, (double) xcache.bytes() / 1073741824.0, failed);
    }
    if (o.expert_cache > 0) {
        std::fprintf(stderr, "strata generate: expert cache %lld slots, %.2f GiB of VRAM; policy is\n",
                     (long long) xcache.slots(), xcache.gib());
        mem_mark("opening the expert cache");
        xcache.set_per_layer_admission(o.expert_cache_per_layer);
        // Round 328 warned here that the GPU hit path was wrong (tokens diverged from a cache-off run from
        // token 0). That fault was fixed long since (native_expert_parity, expert_parity, the grouped kernels'
        // tests), and the warning outlived it (issue #23). What remains is rounding: a GPU expert and the CPU's
        // compute the same quantized expert with different float order, so a near-tie can flip. Measured teacher-
        // forced on 2,557 tokens (bench/results/2026-09-27-cache-parity): 95-98% same top-1, and perplexity equal
        // (on - off = -0.005 +- 0.005 nats). Neither output is more correct than the other.
        std::fprintf(stderr,
                     "strata generate: the GPU computes the experts in the cache; it rounds differently from the CPU,\n"
                     "                 so a reply can differ slightly from a run without the cache (same quality:\n"
                     "                 bench/results/2026-09-27-cache-parity).\n");
        if (o.expert_cache_per_layer) {
            int64_t lo = 0, hi = 0;
            xcache.layer_slot_range(0, lo, hi);
            std::fprintf(stderr, "                 R4.2g PER-LAYER: each layer owns %lld slots (%lld..%lld).\n",
                         (long long) (hi - lo), (long long) lo, (long long) (hi - 1));
        } else if (profile.empty()) {
            std::fprintf(stderr, "                 compulsory-miss (fills with whatever the run routes first).\n");
        } else {
            std::fprintf(stderr, "                 PROFILE, ranked by routing frequency, no eviction.\n");
        }
    }
    // ---- R4.2e: fill the tier from the profile.  This is the only place the plan is applied, and it runs
    // ONCE: with `slots` pairs and `slots` slots the cache is full when this returns, so the decode-time
    // admission finds no room and every non-profiled expert stays a CPU miss.  That is what makes the profile
    // the policy rather than a hint.
    int64_t prefilled = 0;
    if (!profile.empty() && srcp != nullptr) {
        const int64_t want = std::min<int64_t>((int64_t) profile.size(), xcache.slots());
        // mapped reads: advise the next `fill_ahead` pairs so their reads overlap; STRATA_FILL_AHEAD=0 turns it off
        int64_t fill_ahead = 0;
        if (srcp == &src && !src.unbuffered()) {
            const char* v = std::getenv("STRATA_FILL_AHEAD");
            fill_ahead = v == nullptr ? 256 : std::max<int64_t>(0, std::strtol(v, nullptr, 10));
            if (fill_ahead > 0 && !src.advise_pairs(profile.data(), std::min<int64_t>(fill_ahead, want))) fill_ahead = 0;
            std::fprintf(stderr, "strata generate: the profile fill asks for %lld pairs ahead (STRATA_FILL_AHEAD)\n",
                         (long long) fill_ahead);
        }
        const auto fill_t0 = std::chrono::steady_clock::now();
        uint64_t fill_bytes = 0;
        for (int64_t i = 0; i < want; ++i) {
            if (fill_ahead > 0 && i + fill_ahead < want) (void) src.advise_pairs(profile.data() + i + fill_ahead, 1);
            const int32_t slot = xcache.admit(profile[(size_t) i].first, profile[(size_t) i].second);
            if (slot == strata::core::kNotResident) break;
            const uint8_t* b = blob_to_copy(srcp, profile[(size_t) i].first, profile[(size_t) i].second, fill_tmp);
            if (b == nullptr || !xcache.fill_slot_blocking(slot, b, err,
                    (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(profile[(size_t) i].first))) {
                std::fprintf(stderr, "strata generate: the profile fill failed at pair %lld: %s\n",
                             (long long) i, err.c_str());
                return 1;
            }
            fill_bytes += strata::kernels::cpu::expert_layout().blob_bytes(profile[(size_t) i].first);
            ++prefilled;
        }
        const double fill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - fill_t0).count();
        // **AND ONE SLOT IS READ BACK AND COMPARED.**  A residency table that is right about indices and wrong
        // about bytes produces a plausible token, which is this project's most expensive failure mode; the
        // cache's own `verify_slot` is the check and it costs one 1.38 MB D2H at startup.
        if (prefilled > 0 && !xcache.verify_slot(xcache.slot_of(profile[0].first, profile[0].second),
                                blob_to_copy(srcp, profile[0].first, profile[0].second, fill_tmp), err,
                                (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(profile[0].first))) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        mem_mark("the profile fill");
        std::fprintf(stderr, "strata generate: pre-filled %lld of %lld slots from the profile in %.1f s (%.0f MB/s); "
                             "slot 0 verified\n",
                     (long long) prefilled, (long long) want, fill_s,
                     fill_s > 0 ? (double) fill_bytes / 1e6 / fill_s : 0.0);
    }


    for (auto& stp : stages) {
        GpuStage& st = *stp;
        const auto& lay = strata::kernels::cpu::expert_layout();
        const int64_t room = stage_room(st.dev, true, st.ss);
        const strata::core::OnDevice on(st.dev);
        // the expert arena is registered for device copies with CUDA0's context only: this GPU's fills and adaptive
        // swaps copied from it as pageable memory (B70 + 4070, iq3_s: the swaps took 4.5 ms a window against 0.1)
        if (st.dev != 0 && !srcp->register_on(st.dev, err))
            std::fprintf(stderr, "strata generate: layer split, CUDA%d: %s; its copies from the arena are slower\n",
                         st.dev, err.c_str());
        std::vector<int64_t> sized;
        int64_t used = 0;
        for (const auto& pr : st.profile) {
            const int64_t b = native_pack ? ((int64_t) lay.blob_bytes(pr.first) + 255) / 256 * 256 : (int64_t) lay.max_blob;
            if (used + b > room) break;
            used += b;
            sized.push_back((int64_t) lay.blob_bytes(pr.first));
        }
        // its prompt path borrows from this cache, and every stage reads prompts in one chunk size: a cache that holds
        // all its layers' experts in less than the room gets empty slots too, up to what the largest chunk borrows
        // (B70 + 4070, IQ2_XS, K=45: the 4070's 3 layers held 1.3 GiB, and 30K-token prompts read in 3072-token
        // chunks took 25.3 s against 14.6 s on the B70 alone in 32768-token ones)
        if (o.prefill_chunk > 0 && !o.no_prefill_borrow && sized.size() == st.profile.size()) {
            const int64_t top = o.prefill_auto ? std::max<int64_t>(8192, std::min<int64_t>(32768, o.max_context))
                                               : o.prefill_chunk;
            const int64_t want = (int64_t) strata::prefill::Prefill::bytes_needed(g, st.ss, top) * 100 / 85 +
                                 128 * (int64_t) lay.max_blob;
            while (used < want && used + (int64_t) lay.max_blob <= room) {
                used += (int64_t) lay.max_blob;
                sized.push_back((int64_t) lay.max_blob);
            }
        }
        // upstream #841: a stage's cache can fail to open with the VRAM free (the free figure ran high): it is tried
        // again with 90% of the slots, up to 3 times, each try said.  Nothing changes when the first open works.
        bool stage_open = false;
        for (int attempt = 0; !sized.empty(); ++attempt) {
            stage_open = native_pack ? st.cache.open_sized(sized, g.n_layers, g.n_expert, err)
                                     : st.cache.open((int64_t) sized.size(), g.n_layers, g.n_expert, (int64_t) lay.max_blob, err);
            if (stage_open || attempt >= 3) break;
            const size_t keep = sized.size() * 9 / 10;
            std::fprintf(stderr, "strata generate: layer split, CUDA%d expert cache: %s; trying %zu of %zu slots "
                                 "(try %d of 3)\n", st.dev, err.c_str(), keep, sized.size(), attempt + 1);
            if (keep == 0) break;
            st.cache.close();
            sized.resize(keep);
        }
        if (!stage_open) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d expert cache: %s\n", st.dev,
                         sized.empty() ? ("no room (" + std::to_string(room >> 20) + " MiB left after the reserve; on a "
                                          "GPU without memory of its own the RAM the expert arena holds is gone)")
                                             .c_str()
                                       : err.c_str());
            return 1;
        }
        int64_t filled = 0;
        for (const auto& pr : st.profile) {
            if (filled >= st.cache.slots()) break;
            const int32_t slot = st.cache.admit(pr.first, pr.second);
            if (slot == strata::core::kNotResident) break;
            const uint8_t* b = blob_to_copy(srcp, pr.first, pr.second, fill_tmp);
            if (b == nullptr || !st.cache.fill_slot_blocking(slot, b, err, (int64_t) lay.blob_bytes(pr.first))) {
                std::fprintf(stderr, "strata generate: layer split, CUDA%d profile fill failed at pair %lld: %s\n",
                             st.dev, (long long) filled, err.c_str());
                return 1;
            }
            ++filled;
        }
        st.held = filled;
        if (filled == 0 || !st.cache.verify_slot(st.cache.slot_of(st.profile[0].first, st.profile[0].second),
                                                 blob_to_copy(srcp, st.profile[0].first, st.profile[0].second, fill_tmp), err,
                                                 (int64_t) lay.blob_bytes(st.profile[0].first))) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d expert cache: %s\n", st.dev,
                         filled == 0 ? "nothing filled" : err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: layer split: CUDA%d runs layers %lld-%lld, expert cache %lld slots "
                             "(%.2f GiB), %lld of its %zu profiled pairs; slot 0 verified\n",
                     st.dev, (long long) st.lb, (long long) (st.le - 1), (long long) st.cache.slots(), st.cache.gib(),
                     (long long) filled, st.profile.size());
    }
    if (multi_gpu)
        std::fprintf(stderr, "strata generate: layer split: CUDA0 runs layers 0-%lld\n", (long long) (split_at[0] - 1));

    std::array<strata::core::RemoteExperts, 3> remote_experts;
    const bool multi_remote = o.expert_cache_remote[1] > 0 || o.expert_cache_remote[2] > 0;
    if (o.expert_cache_remote[0] > 0) {
        if (o.expert_cache <= 0 || profile.empty() || o.no_pool) {
            std::fprintf(stderr, "strata generate: remote experts need --expert-profile, "
                                 "a CUDA0 expert cache and the expert pool\n");
            return 2;
        }
        std::vector<std::pair<int32_t, int32_t>> ranked = profile;
        if (!stages.empty()) {   // a layer split: CUDA0's share of the profile, then the later stages' pairs no cache holds
            for (auto& st : stages)
                for (const auto& pr : st->profile)
                    if (st->cache.slot_of(pr.first, pr.second) < 0) ranked.push_back(pr);
        }
        if (multi_remote) {
            // The shipped frequency profile names only 8000 of 24576 experts. Once exhausted,
            // fill remaining VRAM from unranked pairs in expert-then-layer order: this spreads
            // the tail across all layers instead of concentrating it on layer zero.
            std::vector<uint8_t> seen((size_t) g.n_layers * (size_t) g.n_expert, 0);
            for (const auto& pair : ranked)
                if (pair.first >= 0 && pair.first < g.n_layers && pair.second >= 0 && pair.second < g.n_expert)
                    seen[(size_t) pair.first * (size_t) g.n_expert + (size_t) pair.second] = 1;
            for (int64_t e = 0; e < g.n_expert; ++e)
                for (int64_t l = 0; l < g.n_layers; ++l)
                    if (!seen[(size_t) l * (size_t) g.n_expert + (size_t) e])
                        ranked.emplace_back((int32_t) l, (int32_t) e);
            std::fprintf(stderr, "strata generate: remote ranking: %zu profiled pairs, "
                                 "%zu other pairs to fill CUDA1..3\n", profile.size(), ranked.size() - profile.size());
        }
        std::array<std::vector<std::pair<int32_t, int32_t>>, 3> by_device;
        if (multi_remote) {
            // Either stripe experts for parallel GPU work, or give each layer one
            // secondary GPU to reduce switches and transfers over shared USB4.
            std::vector<uint8_t> assigned((size_t) g.n_layers * (size_t) g.n_expert, 0);
            const int devices = 1 + (o.expert_cache_remote[1] > 0) + (o.expert_cache_remote[2] > 0);
            int next = 0;
            for (const auto& pair : ranked) {
                if (pair.first < 0 || pair.first >= g.n_layers || pair.second < 0 || pair.second >= g.n_expert ||
                    xcache.slot_of(pair.first, pair.second) >= 0) continue;
                const size_t index = (size_t) pair.first * (size_t) g.n_expert + (size_t) pair.second;
                if (assigned[index]) continue;
                int target = -1;
                if (o.expert_cache_remote_placement == "layer") {
                    target = pair.first % devices;
                    // Other layers' owners may still have room: keep scanning ranks.
                    if (by_device[(size_t) target].size() >=
                        (size_t) o.expert_cache_remote[(size_t) target]) continue;
                } else {
                    for (int i = 0; i < devices; ++i) {
                        const int r = (next + i) % devices;
                        if (by_device[(size_t) r].size() < (size_t) o.expert_cache_remote[(size_t) r]) {
                            target = r;
                            break;
                        }
                    }
                    if (target < 0) break;
                }
                assigned[index] = 1;
                by_device[(size_t) target].push_back(pair);
                next = (target + 1) % devices;
            }
            std::fprintf(stderr, "strata generate: remote ranks %s across %d CUDA devices\n",
                         o.expert_cache_remote_placement == "layer" ? "grouped by layer" : "striped", devices);
        } else {
            by_device[0] = std::move(ranked);
        }
        std::vector<uint8_t> claimed((size_t) g.n_layers * (size_t) g.n_expert, 0);
        for (auto& st : stages)   // a layer split: what a stage's cache holds is no helper's
            for (const auto& pr : st->profile)
                if (st->cache.slot_of(pr.first, pr.second) >= 0)
                    claimed[(size_t) pr.first * (size_t) g.n_expert + (size_t) pr.second] = 1;
        for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0) {
            if (!remote_experts[(size_t) r].open(remote_dev[r], o.expert_cache_remote[(size_t) r],
                     g.n_layers, g.n_expert, by_device[(size_t) r], xcache, *srcp, claimed, err)) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            std::fprintf(stderr, "strata generate: CUDA%d: %lld additional experts, %.2f GiB; "
                                 "results return through pinned host rows\n", remote_dev[r],
                         (long long) remote_experts[(size_t) r].resident(), remote_experts[(size_t) r].gib());
        }
    }

    Drive drive;
    for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0)
        drive.d.remote[drive.d.remote_count++] = &remote_experts[(size_t) r];
    drive.d.hit_cpu_order = o.expert_cache_cpu_order;
    drive.d.split_rows = !o.no_split_rows;
    drive.d.pool = &pool;
    drive.d.src = srcp;
    drive.d.n_expert = g.n_expert;
    drive.d.jobs.resize((size_t) K);
    // ---- R4.2c: THE HIT PATH.  Every one of these is required for `hits_ready()`, which is all-or-nothing on
    // purpose: a half-configured hit path would compute some experts twice and others not at all, and a token
    // built on that is wrong rather than refused.
    void* hit_scratch = nullptr;
    int32_t* d_hit_slot = nullptr;
    int32_t* d_hit_dst = nullptr;
    uint8_t* d_hit_q8 = nullptr;
    float* d_hit_q8_scale = nullptr;   ///< R4.2h: the fp32 activation scales the CPU path also uses
    float* d_hit_out = nullptr;
    if (o.expert_cache > 0 && !o.no_pool) {
        const uint64_t sb = strata::kernels::moe_hit_grouped_scratch_bytes(K, g.n_embd, strata::kernels::cpu::FF);
        if (!strata::gpu::alloc_device(&hit_scratch, (size_t) sb) ||
            !strata::gpu::alloc_device((void**) &d_hit_slot, (size_t) K * sizeof(int32_t)) ||
            !strata::gpu::alloc_device((void**) &d_hit_dst, (size_t) K * sizeof(int32_t)) ||
            !strata::gpu::alloc_device((void**) &d_hit_q8, (size_t) (g.n_embd / 32) * 34) ||
            // R4.2h: the fp32 activation scales.  Without this the GPU's hits use the block's fp16 `d`
            // while the CPU's misses use `ActQ::scale`, which is fp32 - a 4.761e-04 relative disagreement on
            // every chunk, and the reason enabling the cache changed the tokens.
            !strata::gpu::alloc_device((void**) &d_hit_q8_scale, (size_t) (g.n_embd / 32) * sizeof(float)) ||
            !strata::gpu::alloc_device((void**) &d_hit_out, (size_t) K * g.n_embd * 4)) {
            std::fprintf(stderr, "strata generate: the R4 hit path could not allocate its device buffers\n");
            return 1;
        }
        drive.d.cache = &xcache;
        drive.d.cache_stream = main_cs;
        drive.d.cache_base = (const uint8_t*) xcache.device_slot(0);
        drive.d.cache_blob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
        drive.d.cache_slot_off = xcache.slot_offsets();
        drive.d.hit_scratch = hit_scratch;
        drive.d.parts_out = d_parts;
        drive.d.hit_out = d_hit_out;
        drive.d.parts_elems = K * g.n_embd;
        drive.d.mixed = ss.block.mixed;
        drive.d.x_q8_0_hit = d_hit_q8;
        drive.d.x_q8_0_hit_scale = d_hit_q8_scale;
        drive.d.d_slot = d_hit_slot;
        drive.d.d_dst = d_hit_dst;
        drive.d.h_slot.resize((size_t) K);
        strata::gpu::Event* hit_done = nullptr;
        if (!strata::gpu::event_create(&hit_done)) {
            std::fprintf(stderr, "strata generate: the hit path could not create its probe event\n");
            return 1;
        }
        drive.d.hit_done = (void*) hit_done;
        drive.d.hit_poke = !o.no_hit_poke;
        drive.d.h_dst.resize((size_t) K);
        mem_mark("the R4 hit path");
        std::fprintf(stderr, "strata generate: R4 hit path ON - resident experts are computed on the GPU\n");
    }
    // ---- P0.S8: the routing trace.  Only meaningful with the pool running, because the ids arrive through
    // the doorbell that the pool consumes - so `--no-pool` is refused rather than silently producing an empty
    // file that would read as "the router selected nothing".
    // ---- PER-STAGE TIMING.  `--no-capture` only: an event recorded inside a stream capture is silently
    // dropped, so a captured graph cannot carry these events and the numbers would be zeros that read as
    // "every stage is free".  Refusing is the fix.
    if (o.stage_timing) {
        if (!o.no_capture) {
            std::fprintf(stderr, "strata generate: --stage-timing records CUDA events inside the layer path, "
                                 "and an event record inside a stream capture is silently dropped. Pass "
                                 "--no-capture as well.\n");
            return 2;
        }
        if (!strata::core::stage_timing_enable()) {
            std::fprintf(stderr, "strata generate: stage_timing_enable failed\n");
            return 1;
        }
        strata::core::stage_timing_name(0, "gr_read (attn)");
        strata::core::stage_timing_name(1, "attention block");
        strata::core::stage_timing_name(2, "gr_write (attn)");
        strata::core::stage_timing_name(3, "gr_read (ffn)");
        strata::core::stage_timing_name(4, "moe_route");
        strata::core::stage_timing_name(5, "moe_finish");
        strata::core::stage_timing_name(6, "gr_write (ffn)");
        // The GDN block's internals.  It is 36 of the 48 layers, 0.96 ms each, and its entire weight traffic
        // is ~26 MB - so ~0.11 ms at the measured read rate.  ~13 tiny latency-bound launches live in it and
        // a single "attention block" number cannot say which one costs anything.
        strata::core::stage_timing_name(8, "  gdn: quantize x");
        strata::core::stage_timing_name(9, "  gdn: qkv gemv");
        strata::core::stage_timing_name(10, "  gdn: conv+silu");
        strata::core::stage_timing_name(11, "  gdn: l2 norms");
        strata::core::stage_timing_name(12, "  gdn: alpha/beta/gate");
        strata::core::stage_timing_name(13, "  gdn: gdn_step");
        strata::core::stage_timing_name(14, "  gdn: z + out_norm");
        strata::core::stage_timing_name(15, "  gdn: out gemv");
    }
    std::FILE* routing = nullptr;
    if (!o.dump_routing.empty()) {
        if (o.no_pool) {
            std::fprintf(stderr, "strata generate: --dump-routing needs the expert pool; the routed ids reach "
                                 "the host through the doorbell the pool reads. Drop --no-pool.\n");
            return 2;
        }
        routing = std::fopen(o.dump_routing.c_str(), "wb");
        if (routing == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_routing.c_str());
            return 1;
        }
        drive.routing = routing;
    }
    strata::core::PoolFn pool_fn = o.no_pool ? nullptr : &drive_pool;
    // The hit hook rides the same switch as the pool: with no pool there is no `parts` staging to
    // write into, and a hit path with nowhere to write is a wrong token rather than an error.
    strata::core::HitFn hit_fn =
        (o.no_pool || o.expert_cache <= 0) ? nullptr : &strata::core::expert_hit_run;
    void* pool_user = o.no_pool ? nullptr : (void*) &drive;
    std::fprintf(stderr, "strata generate: %d expert-pool workers%s%s\n", pool.workers(),
                 pool.host_works() ? " + the host thread" : "",
                 o.no_pool ? " (UNUSED: --no-pool)" : "");

    // **THE MISALIGNMENT WARNING THAT STOOD HERE IS GONE, BECAUSE THE MISALIGNMENT IS FIXED.**
    //
    // It said the tokens were not the model's, and it was true: `session_loop` handed layer `l`'s expert
    // outputs to layer `l+1`, which multiplied them by layer `l+1`'s router weights (LEDGER L100).  The loop
    // now runs a captured PAIR per layer - `pre[l]` ending with the router and the doorbell, then the CPU
    // pool, then `post[l]` which combines those experts with THAT layer's weights - so layer `l`'s experts meet
    // layer `l`'s routing.  The generated ids changed the moment it landed, which is what a correctness fix
    // looks like from the outside.
    //
    // The cost is real and is recorded rather than hidden: the window for the CPU pool is now whatever GPU
    // work follows the ring inside `pre[l]`, which is the shared expert and nothing else - 0.038 ms against
    // 0.514 ms of CPU work per layer.  A per-layer CPU expert pool cannot be hidden behind a strictly serial
    // residual chain; the CPU term is answered by Phase 3's VRAM expert cache, not by this pipeline.

    // ---- the graphs
    strata::core::SessionGraphs gr;
    if (!o.no_capture && !native_pack) {   // plan v0.3 P6: a native pack runs verify windows only
        if (!strata::core::session_capture(wt, g, ss, d_parts, gr, err, /*split=*/o.gpu_stages)) {
            std::fprintf(stderr, "strata generate: session_capture: %s\n", err.c_str());
            return 1;
        }
    }

    // **`--no-capture` AND THE EXPERTS ARE MUTUALLY EXCLUSIVE, AND SILENTLY SO.**
    //
    // The CPU expert pool is wired into `session_loop` - the host loop around the captured graphs - and
    // `session_token` has no pool hook at all.  So `--no-capture` did not merely change HOW the layers were
    // launched: it ran the whole model with `parts` left at whatever the buffer held, which is ZERO, and the
    // only symptom was `expert blobs 0` in a stats line nobody had to read.  A run that silently omits the
    // routed experts is not a slow measurement of this model, it is a measurement of a different model.
    //
    // Refusing is the fix.  `--no-pool` is the explicit way to say "I want the GPU-only floor".
    if (o.no_capture && !o.no_pool) {
        std::fprintf(stderr,
                     "strata generate: --no-capture runs `session_token`, which has NO CPU expert pool hook, so "
                     "the routed experts would silently contribute nothing. Pass --no-pool as well if the "
                     "GPU-only floor is what you want.\n");
        return 2;
    }
    // The ladder is written by `session_loop`, and `session_token` does not touch the staging buffer at all - so
    // accepting the flag there would produce a file of uninitialised memory, which reads as a wrong answer rather
    // than as a mistake.  `--no-capture` without `--no-pool` is already refused above, so this catches the pair.
    if (o.no_capture && !o.dump_layers.empty()) {
        std::fprintf(stderr,
                     "strata generate: --dump-layers is written by `session_loop`; `--no-capture` runs "
                     "`session_token` instead, which never fills the staging buffer. Drop one of the two.\n");
        return 2;
    }
    if (o.no_capture && !o.dump_halves.empty()) {
        std::fprintf(stderr,
                     "strata generate: --dump-halves is CAPTURED into the layer graphs, so it needs the "
                     "captured path; `--no-capture` never records it. Drop one of the two.\n");
        return 2;
    }

    mem_mark("the expert cache and the graphs");
    std::fprintf(stderr, "strata generate: session is up (engine %s)\n", STRATA_VERSION);
    auto run_head = [&](void* stream) -> bool {
        if (!native_head.loaded())
            return strata::core::lm_head(wt, g, ss.block, d_logits, stream, err);
        return strata::core::lm_head_mix(wt, g, ss.block, stream, err) &&
               native_head.run(ss.block.mixed, d_logits, stream, err);
    };
    float* d_emb = nullptr;
    if (!strata::gpu::alloc_device(&d_emb, (size_t) g.n_embd * 4)) {
        std::fprintf(stderr, "strata generate: the embedding buffer failed\n");
        return 1;
    }
    // **`sample_tokens` TAKES DEVICE POINTERS.**  It is a kernel launch; `logits` and `out` are both read and
    // written on the device.  Passing `logits.data()` - the host vector - faults inside the kernel and the
    // error surfaces at the NEXT synchronising call, which here was the next token's `embed_row`, reporting an
    // illegal access on a weight plane.  Nothing in the parameter names said device.
    int* d_next = nullptr;
    if (!strata::gpu::alloc_device(&d_next, sizeof(int))) {
        std::fprintf(stderr, "strata generate: the sampler output buffer failed\n");
        return 1;
    }

    // **`R` IS BOTH THE INPUT AND THE OUTPUT, SO THE NEW TOKEN'S EMBEDDING HAS TO REPLACE THE OLD RESIDUAL.**
    // At `pos == 0` that is `session_zero`, which is the reference's own initial condition - the embedding
    // broadcast to all `hc` streams.  After that `session_zero` would also wipe the recurrence, so the
    // broadcast is done directly.  Getting this wrong is invisible for exactly one token.
    void* token_stream = o.stream_token ? main_cs : nullptr;
    auto put_input = [&](int64_t tok, int64_t pos) -> bool {
        if (!strata::core::embed_row(wt, g, tok, d_emb, token_stream, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return false;
        }
        if (pos == 0) {
            strata::core::session_zero(ss, g, d_emb, token_stream);
        } else {
            for (int64_t c = 0; c < g.hc; ++c)
                if (!strata::gpu::copy_async(ss.R + (size_t) c * g.n_embd, d_emb, (size_t) g.n_embd * 4, token_stream)) {
                    std::fprintf(stderr, "strata generate: the residual broadcast failed\n");
                    return false;
                }
        }
        return o.stream_token || strata::gpu::device_sync();
    };

    strata::kernels::SamplerParams sp;
    sp.greedy = o.greedy;
    sp.seed = o.seed;
    sp.top_k = o.top_k;
    sp.top_p = o.top_p;
    sp.temperature = o.temperature;
    // what this run actually samples with (the speculative loop below gets the same parameters); serve samples
    // per request instead
    if (!o.serve) {
        if (sp.greedy || sp.temperature <= 0.0f)
            std::fprintf(stderr, "strata generate: sampling greedy\n");
        else
            std::fprintf(stderr, "strata generate: sampling temperature=%g top_k=%d top_p=%g seed=%llu\n",
                         (double) sp.temperature, sp.top_k > 0 && sp.top_k < 64 ? sp.top_k : 64, (double) sp.top_p,
                         (unsigned long long) sp.seed);
    }

    std::FILE* dump = nullptr;
    // A native pack reads the prompt batched (no head there), so its rows start at the last prompt token: the
    // row of position p is p - dump_first.  The per-token loop and the verify windows both write rows.
    const int64_t dump_first = native_pack ? (int64_t) o.tokens.size() - 1 : 0;
    const int64_t dump_positions = (int64_t) o.tokens.size() - 1 - dump_first + o.max_new;
    if (!o.dump_logits.empty()) {
        if (dump_positions > INT32_MAX || n_vocab > INT32_MAX) {
            std::fprintf(stderr, "strata generate: logits dump dimensions exceed int32\n");
            return 2;
        }
        dump = std::fopen(o.dump_logits.c_str(), "wb");
        if (dump == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_logits.c_str());
            return 1;
        }
        // **THE COUNT IS `n_prompt - 1 + max_new`, NOT `n_prompt + max_new`.**  The loop writes one row per
        // position from 0, and it stops once `produced` holds `max_new` tokens - and `produced` only starts
        // receiving at position `n_prompt - 1`.  So a 5-token prompt with `--max-new 6` writes 10 rows, and the
        // header used to claim 11.  A header that describes a different file from the one written is the same
        // class of defect as a self-check that verifies the wrong invariant: anything reading the count instead
        // of the size gets a wrong answer that looks authoritative.  `tools/logits_identical.py` caught it by
        // parsing the header and refusing the file.
        const int32_t n_rows = (int32_t) strata::program::logits_selection::row_count(dump_positions, o.logits_stride);
        const int32_t hdr[2] = {(int32_t) n_vocab, n_rows};
        if (std::fwrite(hdr, sizeof hdr, 1, dump) != 1) {
            std::fprintf(stderr, "strata generate: cannot write logits header\n");
            std::fclose(dump);
            return 1;
        }
    }

    // ---- THE C1 ORACLE: ONE RESIDUAL SNAPSHOT PER LAYER PER POSITION, so the engine can be bisected against
    // `llama-debug`'s `l_last-<il>` node instead of against a single end-to-end perplexity.  The buffer is
    // PINNED because `session_loop` enqueues a device-to-host copy into it after every layer and the transfer
    // would otherwise be staged through a pageable bounce buffer on the critical path.
    std::FILE* layer_dump = nullptr;
    float* layer_stage = nullptr;
    const size_t layer_floats = (size_t) (g.n_layers + 1) * (size_t) g.hc * (size_t) g.n_embd;
    if (!o.dump_layers.empty()) {
        layer_dump = std::fopen(o.dump_layers.c_str(), "wb");
        if (layer_dump == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_layers.c_str());
            return 1;
        }
        if (!strata::gpu::alloc_host((void**) &layer_stage, layer_floats * sizeof(float))) {
            std::fprintf(stderr, "strata generate: cannot pin the layer-dump staging buffer\n");
            return 1;
        }
    }

    // ---- prefill is the DECODE PATH ONE TOKEN AT A TIME, which `phase-2-correct-engine.md:12-13` says is
    // fine here: "process the prompt through the decode-style graphs in small batches; a 19K-token prompt will
    // take minutes".  A real batched prefill is P2.S6's other half and is not this.
    //
    // **THE LOOP IS `feed -> 48 layers -> head -> sample -> feed`, AND THE FIRST GENERATED TOKEN COMES FROM THE
    // LAST *PROMPT* POSITION.**  The first version sampled only on the decode positions, so `produced` was
    // still empty when the first generated position asked for `produced.back()` - an out-of-bounds read on an
    // empty vector.  Teacher forcing below is what makes the distinction unnecessary to special-case: for every
    // position before the last prompt one, the next input is the PROMPT's next token, and after that it is the
    // sampled one.
    std::vector<int64_t> produced;
    double total_ms = 0;
    double prefill_ms = 0;   // positions 0 .. n_prompt-2: prompt tokens that only condition
    const Clock::time_point t_start = Clock::now();
    double ttft_ms = 0;
    const int64_t n_prompt = (int64_t) o.tokens.size();
    int64_t tok = o.tokens[0];

    // ---- THE PURE-GPU MEASUREMENT.  `session_replay` launches all 48 `pre` graphs back to back on one stream
    // with NO host work between them - no doorbell poll, no pool, no parts copy - so what it times is the GPU
    // executing the layer sequence and nothing else.  It had been declared, defined and never called since the
    // day it was written.
    //
    // **THIS IS THE MEASUREMENT THAT SAYS WHETHER THE ENGINE IS HOST-BOUND OR GPU-BOUND**, and the stage table
    // cannot answer it: those events measure the interval between two marks on a stream, which includes every
    // gap where the GPU sat idle waiting for the host to enqueue the next kernel.  In `--no-capture` those gaps
    // are the host's launch latency and they are proportional to the KERNEL COUNT rather than to any work, so
    // the no-capture stage shares are shares of kernel count - which is why the attention block, with the most
    // kernels, looks like 55% of the token there.
    // ---- R0.9: THE PER-STAGE TABLE ON THE CAPTURED GRAPH.
    if (o.gpu_stages) {
        strata::core::doorbell_reset(db);
        double mix = 0, ffn = 0, post = 0;
        if (!strata::core::session_replay_stages(g, 0, 0, ss, gr, main_cs, mix, ffn, post, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        const int reps = 20;
        double t_mix = 0, t_ffn = 0, t_post = 0;
        for (int r = 0; r < reps; ++r) {
            if (!strata::core::session_replay_stages(g, 0, 0, ss, gr, main_cs, mix, ffn, post, err)) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            t_mix += mix;
            t_ffn += ffn;
            t_post += post;
        }
        const double tot = t_mix + t_ffn + t_post;
        std::printf("\nper-stage GPU time on the CAPTURED graph, one token over %lld layers\n",
                    (long long) g.n_layers);
        std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  %6.1f%%\n", "mixer (gr_read+attn+gr_write)",
                    t_mix / reps, t_mix / reps / (double) g.n_layers, 100.0 * t_mix / tot);
        std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  %6.1f%%\n", "ffn front + router",
                    t_ffn / reps, t_ffn / reps / (double) g.n_layers, 100.0 * t_ffn / tot);
        std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  %6.1f%%\n", "post (moe_finish+gr_write)",
                    t_post / reps, t_post / reps / (double) g.n_layers, 100.0 * t_post / tot);
        std::printf("  %-22s %9.3f ms/token\n", "sum of the three", tot / reps);

        // ---- AND THE MIXER BY LAYER KIND, because 36 of the 48 are GDN and 12 are QSA and a total cannot
        // separate them.  Round 309's uncaptured table put GDN at 10.88 ms for 36 layers against QSA's 4.56 for
        // 12, which would make the recurrence the largest single R3 target - and that table had `moe_finish`
        // wrong by 4x, so the ratio is re-derived here from the captured graph rather than inherited.
        {
            // **ACCUMULATED OVER `reps`, NOT MEASURED ONCE AND THEN DIVIDED.**  The first version called the
            // per-layer replay a single time and printed `gdn / reps`, which reported GDN at 0.480 ms/token
            // against a mixer total of 14.039 - a factor of exactly `reps`, and the tell was that
            // 0.480 + 0.220 = 0.700 = 14.039 / 20.
            std::vector<double> acc((size_t) g.n_layers, 0.0), per;
            double f2 = 0, p2 = 0;
            for (int r = 0; r < reps; ++r) {
                if (!strata::core::session_replay_stages_per_layer(g, 0, 0, ss, gr, main_cs, per, f2, p2, err)) {
                    std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                    return 1;
                }
                for (int64_t l = 0; l < g.n_layers; ++l) acc[(size_t) l] += per[(size_t) l];
            }
            double gdn = 0, qsa = 0, worst = 0;
            int64_t ng = 0, nq = 0, worst_l = 0;
            for (int64_t l = 0; l < g.n_layers; ++l) {
                const double v = acc[(size_t) l] / reps;
                if (strata::core::is_qsa_layer(g, l)) { qsa += v; ++nq; }
                else { gdn += v; ++ng; }
                if (v > worst) { worst = v; worst_l = l; }
            }
            std::printf("\n  the mixer by layer kind, averaged over %d runs\n", reps);
            std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  (%lld layers)\n", "GDN layers",
                        gdn, ng ? gdn / (double) ng : 0.0, (long long) ng);
            std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  (%lld layers)\n", "QSA layers",
                        qsa, nq ? qsa / (double) nq : 0.0, (long long) nq);
            std::printf("  %-22s layer %lld at %.3f ms\n", "worst mixer layer", (long long) worst_l, worst);
            std::printf("  %-22s %9.3f ms/token (must equal the mixer above)\n", "GDN + QSA", gdn + qsa);
        }

        // ================================ R0.11: THE FIVE STAGES SEPARATELY ================================
        //
        // Prefixes 1..5 are replayed per layer with the residual restored between them, and consecutive
        // differences are the per-stage times.  **THE CHECK IS THAT THE FIVE SUM TO THE THREE-GRAPH TOTAL** -
        // the same independent-restatement test that caught round 320's divide-by-reps bug, and it is the only
        // reason to believe a table built out of differences.
        {
            std::vector<double> acc5(5, 0.0), per, s5;
            for (int r = 0; r < reps; ++r) {
                if (!strata::core::session_replay_stage_prefixes(g, 0, 0, ss, gr, main_cs, s5, per, err)) {
                    std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                    return 1;
                }
                for (int k = 0; k < 5; ++k) acc5[(size_t) k] += s5[(size_t) k];
            }
            static const char* sn[5] = {"0 gr_read (attn)", "1 attention", "2 gr_write (attn)",
                                        "3 gr_read (ffn)", "4 moe_route (router)"};
            const char* kind[5] = {"GR", "ATTN", "GR", "GR", "ROUTER"};
            double tot5 = 0, gr_ms = 0;
            std::printf("\n  the five stages separately, by differencing prefixes\n");
            std::printf("  %-24s %-8s %10s %10s %8s\n", "stage", "kind", "ms/token", "ms/layer", "share");
            for (int k = 0; k < 5; ++k) {
                const double v = acc5[(size_t) k] / reps;
                tot5 += v;
                if (k != 1 && k != 4) gr_ms += v;
                std::printf("  %-24s %-8s %10.3f %10.4f %7.1f%%\n", sn[k], kind[k], v,
                            v / (double) g.n_layers, 0.0);
            }
            for (int k = 0; k < 5; ++k) {
                const double v = acc5[(size_t) k] / reps;
                (void) v;
            }
            std::printf("  %-24s %-8s %10.3f\n", "sum of the five", "", tot5);
            std::printf("  %-24s %-8s %10.3f   <- R3.3's target is <= 3 ms for all four passes\n",
                        "GR passes (0,2,3)", "GR", gr_ms);
            std::printf("\n  the three-graph total above was %.3f ms/token; the five must account for it.\n",
                        tot / reps);

            // ================================ R3.5c: THE SAME TABLE, A DIFFERENT WAY ================================
            //
            // The differencing table above mixes five graphs per layer, so stage 4's interval carries the launch
            // of the FULL five-stage graph while stage 3's carries a four-stage one.  This sweep launches ONE
            // graph type per layer and nothing else, so that bias cannot exist.  **If the two disagree, the
            // difference IS the bias and this one is right** - and stage 4 is the router, so it is exactly the
            // number that must not be wrong.
            {
                double sweep[6] = {0, 0, 0, 0, 0, 0};
                for (int k = 1; k <= 5; ++k) {
                    double acc = 0, one = 0;
                    for (int r = 0; r < reps; ++r) {
                        if (!strata::core::session_replay_stage_sweep(g, 0, 0, ss, gr, main_cs, k, one, err)) {
                            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                            return 1;
                        }
                        acc += one;
                    }
                    sweep[k] = acc / reps;
                }
                std::printf("\n  the same five stages, by SWEEPING each prefix back to back (no graph switching)\n");
                std::printf("  %-24s %10s %10s %12s\n", "stage", "prefix sweep", "difference", "bias");
                static const char* sn2[5] = {"0 gr_read (attn)", "1 attention", "2 gr_write (attn)",
                                             "3 gr_read (ffn)", "4 moe_route (router)"};
                for (int k = 0; k < 5; ++k) {
                    const double sw = sweep[k + 1] - sweep[k];
                    const double df = acc5[(size_t) k] / reps;
                    std::printf("  %-24s %10.3f %10.3f %11.1f%%\n", sn2[k], sw, df,
                                sw != 0.0 ? 100.0 * (df / sw - 1.0) : 0.0);
                }
                std::printf("  %-24s %10.3f   (full pre, 48 layers)\n", "prefix 5 total", sweep[5]);
            }
        }
        std::printf("\n  compare `--gpu-only-full`, which replays the same work as TWO graphs per layer.  The\n");
        std::printf("  three sum slightly above it because each launch carries the driver's gap.\n");
        strata::core::session_graphs_free(gr);
        strata::core::doorbell_free(db);
        strata::gpu::free(d_next);
        return 0;
    }

    if (o.graph_only) {
        strata::core::doorbell_reset(db);
        // one warm pass so the first launch does not pay for page mapping
        if (!strata::core::session_replay(g, 0, 0, ss, gr, main_cs, err)) {
            std::fprintf(stderr, "strata generate: session_replay warm: %s\n", err.c_str());
            return 1;
        }
        if (!strata::gpu::device_sync()) {
            std::fprintf(stderr, "strata generate: session_replay warm faulted\n");
            return 1;
        }
        const int reps = 20;
        const Clock::time_point t0 = Clock::now();
        for (int r = 0; r < reps; ++r) {
            if (!strata::core::session_replay(g, 0, 0, ss, gr, main_cs, err)) {
                std::fprintf(stderr, "strata generate: session_replay: %s\n", err.c_str());
                return 1;
            }
        }
        if (!strata::gpu::device_sync()) {
            std::fprintf(stderr, "strata generate: session_replay faulted: %s\n", strata::gpu::last_error());
            return 1;
        }
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / (double) reps;
        std::printf("pre graphs only   %8.2f ms per token over %lld layers  ->  %.2f tok/s of GPU work\n", ms,
                    (long long) g.n_layers, ms > 0 ? 1000.0 / ms : 0.0);
        std::printf("                  %8.3f ms per layer\n", ms / (double) g.n_layers);
        return 0;
    }

    // ---- R0.3: THE TRUE PER-TOKEN GPU FLOOR.
    //
    // `--graph-only` above launches ONLY `gr.execs[l]`, the `pre` graphs.  It omits the 48 `post` graphs - the
    // shared expert, the combine and the second `gr_write` - and the LM head.  Everything this project published
    // as "39.8 ms pure GPU" came from that loop while being described as the whole GPU, which also made the
    // "host's share = 49.4 - 39.8 = 9.6 ms" figure wrong by however much the missing work costs.  See
    // Memory/ERRORS.md A4/A5.
    //
    // This flag is what "pure GPU" has to mean, and it replaces that number everywhere.  No pool runs, so
    // `parts` keeps whatever the buffer holds and the timing is GPU work alone.
    if (o.gpu_only_full) {
        strata::core::doorbell_reset(db);
        const int reps = 20;
        double ms_layers = 0, ms_head = 0;
        for (int r = -1; r < reps; ++r) {          // r == -1 is the warm pass, not counted
            const Clock::time_point t0 = Clock::now();
            if (!strata::core::session_replay_full(g, 0, 0, ss, gr, main_cs, err)) {
                std::fprintf(stderr, "strata generate: session_replay_full: %s\n", err.c_str());
                return 1;
            }
            // The sync is INSIDE the interval on purpose: it is the wait for the GPU, so t1 - t0 is GPU time.
            if (!strata::gpu::stream_sync(main_cs)) {
                std::fprintf(stderr, "strata generate: gpu-only-full layers faulted: %s\n",
                             strata::gpu::last_error());
                return 1;
            }
            const Clock::time_point t1 = Clock::now();
            if (!run_head(main_cs)) {
                std::fprintf(stderr, "strata generate: gpu-only-full lm_head: %s\n", err.c_str());
                return 1;
            }
            if (!strata::gpu::stream_sync(main_cs)) {
                std::fprintf(stderr, "strata generate: gpu-only-full head faulted: %s\n",
                             strata::gpu::last_error());
                return 1;
            }
            const Clock::time_point t2 = Clock::now();
            if (r < 0) continue;
            ms_layers += std::chrono::duration<double, std::milli>(t1 - t0).count();
            ms_head += std::chrono::duration<double, std::milli>(t2 - t1).count();
        }
        ms_layers /= (double) reps;
        ms_head /= (double) reps;
        const double ms = ms_layers + ms_head;
        std::printf("GPU floor pre+post+head %7.2f ms per token  ->  %.2f tok/s of GPU work\n", ms,
                    ms > 0 ? 1000.0 / ms : 0.0);
        std::printf("                  %8.3f ms layers (%lld x pre+post)\n", ms_layers, (long long) g.n_layers);
        std::printf("                  %8.3f ms per layer\n", ms_layers / (double) g.n_layers);
        std::printf("                  %8.3f ms LM head\n", ms_head);
        return 0;
    }

    // ---- **THE TOKEN PATH ALLOCATES NOTHING (P2.T10, review finding H3).**
    //
    // `session_loop` used to allocate its pinned staging buffer, its probe event and its host pin ON EVERY
    // TOKEN, and `cudaFreeHost` at the end of each call implicitly synchronises the device - so every token
    // finished with a device-wide sync nobody asked for.  The scratch is created once here and reused; it also
    // owns the host pin for the whole session rather than taking and releasing it per token.
    strata::core::SessionLoopScratch loop_scratch;
    struct ScratchFree {
        strata::core::SessionLoopScratch* p;
        ~ScratchFree() { if (p != nullptr) p->free(); }
    } scratch_free{&loop_scratch};
    // Initialised unconditionally, including under --no-pool: the loop validates the scratch it is handed, so
    // passing a default-constructed one is an error rather than a fallback.  (It was, and the guard caught it -
    // which is the point of the guard.)  One allocation at setup either way.
    if (!loop_scratch.init((size_t) K * g.n_embd * 4, err)) {
        std::fprintf(stderr, "strata generate: %s\n", err.c_str());
        return 1;
    }
    // Plan v0.3 P3: the whole token as ONE graph whenever nothing needs a host step between the ring and post[l]
    // (the VRAM expert tier and the per-layer dumps do).  `--no-token-graph` keeps two graphs per layer.
    strata::core::TokenGraph tgraph;
    struct TokenGraphFree {
        strata::core::TokenGraph* p;
        ~TokenGraphFree() { strata::core::token_graph_free(*p); }
    } tgraph_free{&tgraph};
    // Plan v0.3 P4: with a PROFILE-filled cache the residency is static, so the hit decision moves onto the
    // device and the token graph keeps it.  (A cache filled on demand still needs the per-layer host path.)
    std::vector<int32_t> host_res;
    int32_t* d_res = nullptr;
    int32_t* d_hit_count = nullptr;
    strata::core::TokenHits thits;
    const bool graph_hits = hit_fn != nullptr && !profile.empty() && !o.no_pool;
    if (graph_hits && !o.no_capture && !o.no_token_graph && layer_dump == nullptr && half_dump == nullptr) {
        host_res.assign((size_t) (g.n_layers * g.n_expert), strata::core::kNotResident);
        int64_t resident = 0;
        for (int64_t l = 0; l < g.n_layers; ++l)
            for (int64_t e = 0; e < g.n_expert; ++e) {
                const int st = multi_gpu ? stage_of(l) : 0;
                const int32_t slot = st > 0 ? stages[(size_t) st - 1]->cache.slot_of(l, e) : xcache.slot_of(l, e);
                host_res[(size_t) (l * g.n_expert + e)] = slot;
                if (slot != strata::core::kNotResident) ++resident;
            }
        if (!strata::gpu::alloc_device((void**) &d_res, host_res.size() * sizeof(int32_t)) ||
            !strata::gpu::alloc_device((void**) &d_hit_count, sizeof(int32_t)) ||
            !strata::gpu::copy(d_res, host_res.data(), host_res.size() * sizeof(int32_t))) {
            std::fprintf(stderr, "strata generate: the device residency table could not be staged\n");
            return 1;
        }
        thits.d_res = d_res;
        thits.n_expert = g.n_expert;
        // a file-backed arena (STRATA_ARENA_MMAP): the experts a GPU holds are handed back first
        // (STRATA_ARENA_RELEASE=0 keeps them): the fill read the whole arena, and left mapped and referenced it crowds
        // every other allocation into swap.  The ones no GPU holds are those the CPU pool and the prompt path will
        // read: start reading them now instead of faulting them in 4 KB at a time mid-request
        {
            static const bool keep = std::getenv("STRATA_ARENA_RELEASE") && std::getenv("STRATA_ARENA_RELEASE")[0] == '0';
            uint64_t released = 0;
            if (!keep)
                for (size_t i = 0; i < host_res.size(); ++i)
                    if (host_res[i] != strata::core::kNotResident)
                        released += srcp->release((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);
            if (released > 0)
                std::fprintf(stderr, "strata generate: expert arena: %.2f GiB of VRAM-held experts handed back to the OS\n",
                             (double) released / (1024.0 * 1024.0 * 1024.0));
            for (size_t i = 0; i < host_res.size(); ++i)
                if (host_res[i] == strata::core::kNotResident)
                    srcp->prefetch((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);
        }
        for (auto& st : stages) {   // layer split across GPUs: the same table on every device
            const strata::core::OnDevice on(st->dev);
            if (!strata::gpu::alloc_device((void**) &st->d_res, host_res.size() * sizeof(int32_t)) ||
                !strata::gpu::copy(st->d_res, host_res.data(), host_res.size() * sizeof(int32_t))) {
                std::fprintf(stderr, "strata generate: layer split: CUDA%d residency table failed\n", st->dev);
                return 1;
            }
        }
        thits.cache_base = drive.d.cache_base;
        thits.blob = drive.d.cache_blob;
        thits.d_slot = drive.d.d_slot;
        thits.d_dst = drive.d.d_dst;
        thits.d_count = d_hit_count;
        thits.x_q8 = drive.d.x_q8_0_hit;
        thits.x_scale = drive.d.x_q8_0_hit_scale;
        thits.scratch = drive.d.hit_scratch;
        thits.hit_out = drive.d.hit_out;
        drive.d.host_res = host_res.data();
        std::fprintf(stderr, "strata generate: token graph hit path: %lld resident experts, decided on the device\n",
                     (long long) resident);
    }
    if (!o.no_capture && !o.no_token_graph && layer_dump == nullptr && half_dump == nullptr &&
        (hit_fn == nullptr || thits.on()) && !native_pack) {
        if (!strata::core::session_capture_token(wt, g, ss, d_parts, loop_scratch.y_miss, loop_scratch.parts_bytes,
                                                 tgraph, err, thits.on() ? &thits : nullptr)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: token graph captured (48 layers, one launch per token)\n");
    }

    // ================================ WHERE THE HOST TERM GOES, PER TOKEN ================================
    //
    // **`--gpu-only-full` MEASURES THE 48 LAYER GRAPHS AND THE LM HEAD AND NOTHING ELSE.**  It never enters
    // this loop, so it does not run `ple_stage_token`, `embed_row`, the whole-vocabulary logits readback, the
    // NaN scan or the sampler - and `--no-pool --stats` against that floor was being read as "the per-layer
    // round trip costs 12.4 ms" when an unknown part of it is per TOKEN, not per layer.  That is the same error
    // the review catalogued as A4/A5, one level down: a difference between two measurements attributed to a
    // mechanism that neither of them isolates.
    //
    // Six accumulators, because the six have different fixes.  Reported in `--stats` as ms/token.  **The
    // boundary after the layer loop is the one that matters**: without it the head's interval swallows all 48
    // layers and the report reads as "head = 50 ms", which is not a thing that can happen to 1.4 ms of GPU
    // work.  That is not hypothetical - it is what the first version of this printed.
    double ms_ple = 0, ms_embed = 0, ms_layers = 0, ms_head = 0, ms_readback = 0, ms_sample = 0;
    int64_t phase_tokens = 0;

    // ================================ plan v0.3 P8: THE PERSISTENT ENGINE (--serve) ================================
    //
    // The weights, the expert arena and the VRAM tier load once; then requests arrive on stdin, one per line,
    //
    //     GEN <max_new> <id,id,...>
    //
    // and each generated token is written to stdout as `T <id>` as soon as its verify window is done, followed by
    //
    //     DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <stop|length|cancel> <drafts accepted>
    //          <drafts offered> <prompt tokens reused> ... <prompt tokens read>   (see the DONE line below; #471)
    //
    // Before that, `RESUME <n>` (n prompt tokens are not read again), `PP <position> <prompt_tokens> <ms> <tok/s>`
    // after every prompt chunk, and `REUSED <n>` once the prompt is read.  (`ERR <message>` instead when a request
    // cannot run; `STOP` ends the running request at its next step; `QUIT` ends the process.)  A request continues
    // from the live session or the longest conversation checkpoint its prompt starts with (see ConvCheckpoint),
    // otherwise from an empty sequence (`session_zero`); the rest of the prompt goes through the batched prompt path
    // and its last token through the first verify window - the path all three model files share.  Decoding is greedy.
    // The expert-cache slots (from the end of the cache) that hold the prompt path's buffers for a chunk, and the
    // bytes from the first of them to the end.
    strata::prefill::Prefill::set_expert_group(o.prefill_experts);
    // the share of expert bytes the arena could pin (sizes the prompt path's streamed ring and its lend cap)
    if (srcp != nullptr && o.prefill_chunk > 0) {
        uint64_t pinned = 0, total = 0;
        const auto& lay = strata::kernels::cpu::expert_layout();
        for (int64_t l = 0; l < g.n_layers; ++l)
            for (int64_t e = 0; e < g.n_expert; ++e) {
                const uint64_t b = lay.blob_bytes(l);
                total += b;
                if (srcp->pinned(l, e)) pinned += b;
            }
        strata::prefill::Prefill::set_pinned_share(total ? (double) pinned / (double) total : 1.0);
    }
    // (of a cache and its session on device `dev`: CUDA0's, or a layer split stage's - each stage's prompt path
    // borrows from its own GPU's cache, with the chunk every stage shares; -1: the current device)
    auto lend_slots_of = [&](const strata::core::ExpertCache& c, const strata::core::SessionState& s, int dev,
                             int64_t chunk) -> int64_t {
        const strata::core::OnDevice on(dev);   // the streamed ring is sized from that GPU's memory
        const uint64_t need = strata::prefill::Prefill::bytes_needed(g, s, chunk);
        const int64_t blob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
        int64_t k = (int64_t) ((need + (uint64_t) blob - 1) / (uint64_t) blob);
        if (c.slot_offsets() != nullptr) {   // sized slots: take slots from the end until they hold `need`
            k = 0;
            while (k < c.slots() && (uint64_t) (c.bytes() - (int64_t) c.slot_offsets()[c.slots() - k]) < need) ++k;
        }
        return k;
    };
    auto lend_bytes_of = [&](const strata::core::ExpertCache& c, int32_t first) -> uint64_t {
        return c.slot_offsets() ? (uint64_t) (c.bytes() - (int64_t) c.slot_offsets()[first])
                                : (uint64_t) (c.slots() - first) * (uint64_t) strata::kernels::cpu::expert_layout().max_blob;
    };
    auto lend_slots = [&](int64_t c) -> int64_t { return lend_slots_of(xcache, ss, -1, c); };
    auto lend_bytes = [&](int32_t first) -> uint64_t { return lend_bytes_of(xcache, first); };
    // every later stage's cache can lend a chunk of c tokens (`fits` its slot count)
    auto stages_lend = [&](int64_t c, const auto& fits) -> bool {
        for (const auto& st : stages)
            if (!fits(st->cache, lend_slots_of(st->cache, st->ss, st->dev, c))) return false;
        return true;
    };
    // --prefill auto: the slots a chunk of c tokens borrows, 0 when it does not fit.  At 8192-token chunks nearly
    // every expert streams anyway, so a lent slot costs little: 90% when the copies are DMA from pinned RAM (Q2_0
    // 8192 + a 384-slot ring: 1283 tok/s), 85% when host copies are the limit (lending more only streams more
    // through them).  STRATA_PREFILL_LEND_PCT overrides (tuning).  A chunk that does not fit with the streamed ring
    // of the pinned-share rule takes a 96-slot ring instead (upstream #583): fewer chunks stream far fewer experts,
    // and the bigger ring does not buy that back - on the B70 at STRATA_VRAM_LIMIT_MIB=8192, IQ3_S, interleaved:
    // 8K prompt 6144/default ring 532-817 tok/s, 8192/96 717-1161 (6 pairs); 32K 4096/default 565-617, 8192/96
    // 1046-1293 (3 pairs); 4K 405-653, 283-712 (3 pairs).  STRATA_PREFILL_RING still wins.
    auto auto_slots = [&](int64_t c) -> int64_t {
        const int64_t pct = [] {
            const char* v = std::getenv("STRATA_PREFILL_LEND_PCT");
            return v ? (int64_t) std::strtol(v, nullptr, 10)
                     : (int64_t) (strata::prefill::Prefill::pinned_share() >= 0.9 ? 90 : 85);
        }();
        auto fits = [&](const strata::core::ExpertCache& x, int64_t k) {
            return k + 128 <= x.slots() && k * 100 <= pct * x.slots();
        };
        strata::prefill::Prefill::set_ring_override(0);
        if (const int64_t k = lend_slots(c); fits(xcache, k) && stages_lend(c, fits)) return k;
        strata::prefill::Prefill::set_ring_override(96);
        if (const int64_t k = lend_slots(c); fits(xcache, k) && stages_lend(c, fits)) return k;
        strata::prefill::Prefill::set_ring_override(0);
        return (int64_t) 0;
    };
    // The prompt path's chunk and the slots it borrows for its buffers: the requested chunk halved until it fits,
    // or with --prefill auto the largest of kAutoChunks whose buffers take at most kAutoLendPct % of the slots (a
    // lent slot's expert is streamed during the prompt and refilled after it; measured on a 12 GB card, 32K Q2_0
    // prompt: 4096 791 tok/s, 6144 878, 8192 973 with 69% of the slots lent).  A request lends only what its own
    // prompt needs (Prefill::relayout), so a big chunk costs short prompts nothing.  0 = none fits.
    auto plan_lend = [&](int64_t& chunk) -> int64_t {
        // 32768 and 16384 (upstream ae652b2; opt-in there, 3e31eea, as other chunks rounded differently on CUDA):
        // every chunk streams again each expert it routes to, so on the B70 a 26,292-token IQ3_S prompt read at
        // 1,080 tok/s in 8192-token chunks (50,356 experts streamed), 1,303 in 16384 and 1,545 in one chunk (17,728),
        // the same logits.  They take the lend rule below like the others, so a smaller cache picks a smaller chunk,
        // and none goes past --max-context.  STRATA_PREFILL_AUTO_MAX lowers the ceiling (8192: as before).
        static constexpr int64_t kAutoChunks[] = {32768, 16384, 8192, 6144, 4096, 3072, 2048, 1024, 512, 256};
        static const int64_t auto_max = [] {
            const char* v = std::getenv("STRATA_PREFILL_AUTO_MAX");
            return v ? (int64_t) std::strtoll(v, nullptr, 10) : (int64_t) 32768;
        }();
        if (o.prefill_auto) {
            for (const int64_t c : kAutoChunks) {
                if (c > auto_max || (c > 8192 && c > o.max_context)) continue;
                if (const int64_t k = auto_slots(c); k > 0) { chunk = c; return k; }
            }
            return 0;
        }
        auto fits = [](const strata::core::ExpertCache& x, int64_t k) { return k + 128 <= x.slots(); };
        for (int64_t c = chunk; c >= 256; c /= 2) {
            const int64_t k = lend_slots(c);
            if (fits(xcache, k) && stages_lend(c, fits)) { chunk = c; return k; }
        }
        return 0;
    };
    if ((o.resident_cpu_experts || o.resident_gib != 0) && srcp == &src) {
        const uint64_t budget = o.resident_gib < 0 ? strata::core::FileExpertSource::kResidentWhatFits
                                : (uint64_t) (o.resident_gib * 1073741824.0);
        int64_t lend_from = -1;
        if (o.prefill_chunk > 0 && !o.no_prefill_borrow && d_res && xcache.slots() > 0) {
            int64_t chunk = o.prefill_chunk;
            const int64_t slots = plan_lend(chunk);
            if (slots > 0) lend_from = xcache.slots() - slots;
        }
        // with a layer split the RAM tier serves every stage's layers: ranked by the whole profile, without the experts
        // the later stages' caches hold (CUDA0's `profile` is its own layers' only; B70 + 4070, UD-Q4_K_XL: 26 GiB
        // of the 66 GiB budget were filled, and the later stage's misses were read from the SSD)
        std::vector<std::pair<int32_t, int32_t>> other_gpus;
        for (const auto& st : stages)
            other_gpus.insert(other_gpus.end(), st->profile.begin(), st->profile.begin() + st->held);
        if (!src.pin_cache_complement(xcache, err, o.resident_pin, other_gpus, lend_from, 8ull << 30, budget,
                                      stages.empty() ? &profile : &profile_all) ||
            (o.adapt_every > 0 && o.adapt_swaps > 0 && !src.reserve_exchanges(o.adapt_swaps, err))) {
            std::fprintf(stderr, "strata generate: CPU expert residency: %s\n", err.c_str()); return 1;
        }
        // #669 / #765: a big chunk lends many cache slots to the prompt path, and each lent slot's expert is read back
        // from the SSD whenever the RAM could not keep its copy (31 GB: auto:32768 read prompts ~3x slower than auto).
        // Said, never capped: --prefill is the user's, and with the RAM for them it is the faster one (+21-35%).
        if (lend_from >= 0 && o.prefill_chunk > 8192 && xcache.slots() > lend_from &&
            src.resident_lent_slots() < xcache.slots() - lend_from)
            std::fprintf(stderr, "strata generate: WARNING: a prompt chunk of %lld tokens lends %lld cache slots to the "
                                 "prompt path, but only %lld of their experts fit in RAM: the others are read from the "
                                 "SSD on every chunk, and prompts can read ~3x slower than with --prefill auto (#669). "
                                 "A smaller --prefill, or auto, keeps them all in RAM\n",
                         (long long) o.prefill_chunk, (long long) (xcache.slots() - lend_from),
                         (long long) src.resident_lent_slots());
        std::fprintf(stderr, "strata generate: resident CPU experts: %.2f GiB, %lld experts\n",
                     (double) src.resident_bytes() / 1073741824.0, (long long) src.resident_count());
        mem_mark("the resident experts");
    }
    if (o.serve) {
        if (o.spec < 2 || o.prefill_chunk <= 0 ||
            (graph_hits && (thits.d_res == nullptr || host_res.empty()))) {
            std::fprintf(stderr, "strata serve: needs --spec T and --prefill CHUNK (and a fillable "
                                 "--expert-cache; the graphed hit path additionally needs --expert-profile P)\n");
            return 2;
        }
        // --mtp is optional (upstream 3216d271): without it the drafts come from the suffix / prompt-lookup drafter
        // only, or one token a round, and every token is still verified; the draft layer's VRAM goes to the expert
        // cache.  The parked and saved conversations carry the draft layer's K/V, so they are off without it.
        const bool use_mtp = !o.mtp.empty();
        if (!use_mtp && ((o.prompt_cache > 0 && o.conversation_cache_mib > 0) || !o.conversation_save.empty())) {
            std::fprintf(stderr, "strata serve: without --mtp the conversation cache's parked conversations and "
                                 "--conversation-save are off (a parked conversation carries the draft layer's K/V)\n");
            o.conversation_save.clear();
        }
        strata::prefill::Prefill sp;
        void* borrow = nullptr;
        uint64_t borrow_bytes = 0;
        int32_t lend_first = -1;          // the first slot the prompt path may borrow (its largest chunk)
        int32_t lend_first_now = -1;      // where its buffers are laid out now
        // a cache too small to lend the prompt path its buffers would make it allocate them on top - on a card the
        // cache already filled to its reserve, that is the over-subscription the auto sizing avoids - so the
        // prompt chunk is halved until its buffers fit in the lendable slots (a smaller chunk only reads slower)
        if (!o.no_prefill_borrow && d_res != nullptr) {
            int64_t chunk = o.prefill_chunk;
            if (const int64_t k = plan_lend(chunk); k > 0) {
                if (o.prefill_auto)
                    std::fprintf(stderr, "strata serve: prompt chunk auto: %lld tokens\n", (long long) chunk);
                else if (chunk != o.prefill_chunk)
                    std::fprintf(stderr, "strata serve: prompt chunk %lld -> %lld tokens so its buffers fit in the "
                                         "expert cache\n", (long long) o.prefill_chunk, (long long) chunk);
                o.prefill_chunk = chunk;
                lend_first = (int32_t) (xcache.slots() - k);
                lend_first_now = lend_first;
                borrow = xcache.device_slot(lend_first);
                borrow_bytes = xcache.slot_offsets()
                                   ? (uint64_t) (xcache.bytes() - (int64_t) xcache.slot_offsets()[lend_first])
                                   : (uint64_t) k * (uint64_t) strata::kernels::cpu::expert_layout().max_blob;
            } else if (o.prefill_auto) {
                o.prefill_chunk = 1024;   // nothing lendable: small buffers of its own
            }
        } else if (o.prefill_auto && d_res == nullptr) {
            o.prefill_chunk = 1024;       // #85: no expert cache at all (a full 8 GB card): small buffers of its own
        }
        if (borrow != nullptr)
            std::fprintf(stderr, "strata serve: the prompt path borrows %lld cache slots (%.2f GiB)\n",
                         (long long) (xcache.slots() - lend_first), (double) borrow_bytes / 1073741824.0);
        else
            std::fprintf(stderr, "strata serve: the prompt path allocates its own buffers (too few cache slots to borrow)\n");
        // layer split across GPUs: a prompt path per stage, each handing its chunk's rows to the next; with CUDA0's
        // loan each borrows the chunk's buffers from the end of its own GPU's cache (plan_lend chose a chunk they fit)
        std::vector<int32_t> st_lend_first(stages.size(), -1), st_lend_first_now(stages.size(), -1);
        for (size_t i = 0; i < stages.size(); ++i) {
            GpuStage& st = *stages[i];
            st.sp.set_stage(st.lb, i + 1 < stages.size() ? st.le : -1, i + 1 < stages.size() ? &stages[i + 1]->sp : nullptr);
            void* st_borrow = nullptr;
            uint64_t st_borrow_bytes = 0;
            if (lend_first >= 0) {
                const int64_t k = lend_slots_of(st.cache, st.ss, st.dev, o.prefill_chunk);
                st_lend_first[i] = st_lend_first_now[i] = (int32_t) (st.cache.slots() - k);
                st_borrow = st.cache.device_slot(st_lend_first[i]);
                st_borrow_bytes = lend_bytes_of(st.cache, st_lend_first[i]);
                std::fprintf(stderr, "strata serve: layer split, CUDA%d: the prompt path borrows %lld cache slots "
                                     "(%.2f GiB)\n", st.dev, (long long) k, (double) st_borrow_bytes / 1073741824.0);
            }
            const strata::core::OnDevice on(st.dev);
            if (!st.sp.init(st.wt, g, st.ss, srcp, &st.cache, host_res.data(), o.prefill_chunk, (void*) st.stream, err,
                            st_borrow, st_borrow_bytes)) {
                std::fprintf(stderr, "strata serve: layer split, CUDA%d prompt path: %s\n", st.dev, err.c_str());
                return 1;
            }
        }
        if (multi_gpu) sp.set_stage(0, split_at[0], &stages[0]->sp);
        if (!sp.init(wt, g, ss, srcp, &xcache, host_res.data(), o.prefill_chunk, main_cs, err, borrow, borrow_bytes)) {
            std::fprintf(stderr, "strata serve: %s\n", err.c_str());
            if (err.find("fit") != std::string::npos)   // #85: say what frees VRAM
                std::fprintf(stderr, "strata serve: the GPU has too little free VRAM for the prompt path: turn images "
                                     "off (setup: --vision no), close other programs using the GPU, use a shorter "
                                     "context, or read prompts in smaller chunks (--prefill 512)\n");
            return 1;
        }
        mem_mark("the head and the prompt path");
        // the penalty-history buffer: one row per verify-window row (`penalty_rows`), each the last
        // `penalty_last_n` tokens that row's pick follows, -1 padded in front.  Allocated once at the cap for
        // the widest window; a request without penalties gets a null buffer and takes the byte-for-byte
        // neutral path (no upload, no buffer handed to the sampler).
        constexpr int kPenaltyWindowCap = 4096;
        constexpr size_t kHistSlots = (size_t) kPenaltyWindowCap * (size_t) strata::kernels::kVerifyMaxT;
        int32_t* d_hist = nullptr;
        std::vector<int32_t> hist_stage(kHistSlots, -1);
        const int hist_dev = last_st ? last_st->dev : -1;   // with the head: the last stage's device
        if (const strata::core::OnDevice on_h(hist_dev); !strata::gpu::alloc_device(&d_hist, kHistSlots * sizeof(int32_t))) {
            std::fprintf(stderr, "strata serve: the penalty-history allocation failed\n");
            return 1;
        }
        strata::core::Verifier ver;
        strata::core::VerifyHits vh;
        vh.d_res = thits.d_res;
        vh.cache_base = thits.cache_base;
        vh.blob = thits.blob;
        vh.slot_off = xcache.slot_offsets();   // E-6: the device plan's pointers
        vh.n_slots = xcache.slots();
        vh.h_res = host_res.empty() ? nullptr : host_res.data();   // every expert resident: windows without the host
        // Layer split: `ver` runs layers [0, K1) and hands its residual to the next stage's verifier, and so on; the
        // last runs the head.  The hand-offs are host USM: one buffer between two stages on one GPU, and one on each
        // side between stages on two (a context's host USM is not visible to another GPU; Verifier::run copies).
        // (--split-device 0: the second stage on this GPU, sharing its weights, session and cache - the A/B.)
        strata::core::Verifier ver_same;
        SplitDrive split_drive;
        auto stage_ver = [&](int st) -> strata::core::Verifier& {
            return st == 0 ? ver : split_same ? ver_same : stages[(size_t) st - 1]->ver;
        };
        auto pcie_num_of = [](double f) { return std::max(0, std::min(256, (int) (f * 256.0 + 0.5))); };
        const int n_stages = split_devs.empty() ? 1 : (int) split_at.size() + 1;
        if (n_stages > 1) {
            const size_t hb = (size_t) strata::kernels::kVerifyMaxT *
                              (size_t) strata::core::Verifier::handoff_floats(g) * sizeof(float);
            auto dev_of = [&](int st) { return st == 0 || split_same ? 0 : stages[(size_t) st - 1]->dev; };
            auto hand_alloc = [&](int dev) -> float* {
                const strata::core::OnDevice on(dev);
                float* hh = nullptr;
                if (!strata::gpu::alloc_host((void**) &hh, hb)) return nullptr;
                std::memset(hh, 0, hb);
                return hh;
            };
            // hand_out[st]: what stage st writes; hand_in[st]: what stage st + 1 reads
            std::vector<float*> hand_out((size_t) n_stages - 1, nullptr), hand_in((size_t) n_stages - 1, nullptr);
            for (int st = 0; st + 1 < n_stages; ++st) {
                hand_out[(size_t) st] = hand_alloc(dev_of(st));
                hand_in[(size_t) st] =
                    dev_of(st + 1) == dev_of(st) ? hand_out[(size_t) st] : hand_alloc(dev_of(st + 1));
                if (!hand_out[(size_t) st] || !hand_in[(size_t) st]) {
                    std::fprintf(stderr, "strata serve: the layer-split hand-off allocation failed\n");
                    return 1;
                }
            }
            split_drive.base = &drive;
            split_drive.n = n_stages;
            for (int st = 0; st < n_stages; ++st) {
                stage_ver(st).set_stage(st == 0 ? 0 : split_at[(size_t) st - 1], st + 1 < n_stages ? split_at[(size_t) st] : -1,
                                        st == 0 ? nullptr : hand_in[(size_t) st - 1],
                                        st + 1 < n_stages ? hand_out[(size_t) st] : nullptr);
                split_drive.end[st] = st + 1 < n_stages ? split_at[(size_t) st] : g.n_layers;
                split_drive.cache_base[st] = drive.d.cache_base;
                split_drive.cache_slot_off[st] = drive.d.cache_slot_off;
                split_drive.pcie_num[st] = pcie_num_of(o.pcie_frac);
            }
            for (int st = 1; st < n_stages; ++st) {
                bool ok_s = false;
                if (split_same) {
                    ok_s = ver_same.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr,
                                         std::max(o.spec, o.batch), err);
                } else {
                    GpuStage& gs = *stages[(size_t) st - 1];
                    const strata::core::OnDevice on(gs.dev);
                    strata::core::VerifyHits vs;
                    vs.d_res = gs.d_res;
                    vs.cache_base = gs.cache.device_slot(0);
                    vs.blob = thits.blob;
                    vs.slot_off = gs.cache.slot_offsets();
                    vs.n_slots = gs.cache.slots();
                    ok_s = gs.ver.init(gs.wt, g, gs.ss, vs, gs.head.loaded() ? &gs.head : nullptr,
                                       std::max(o.spec, o.batch), err);
                    split_drive.cache_base[st] = gs.cache.device_slot(0);
                    split_drive.cache_slot_off[st] = gs.cache.slot_offsets();
                    split_drive.pcie_num[st] = pcie_num_of(gs.pcie_frac);
                }
                if (!ok_s) {
                    std::fprintf(stderr, "strata serve: layer split, stage %d: %s\n", st + 1, err.c_str());
                    return 1;
                }
            }
            for (int st = 0; st + 1 < n_stages; ++st) stage_ver(st).set_next(&stage_ver(st + 1), &split_drive);
            std::string plan_s = "0-" + std::to_string(split_at[0] - 1) + " (CUDA0)";
            for (int st = 1; st < n_stages; ++st)
                plan_s += ", " + std::to_string(split_at[(size_t) st - 1]) + "-" + std::to_string(split_drive.end[st] - 1) +
                          " (CUDA" + std::to_string(split_same ? 0 : stages[(size_t) st - 1]->dev) + ")";
            std::fprintf(stderr, "strata serve: layer split: layers %s, one hand-off per window\n", plan_s.c_str());
        }
        if (!ver.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, std::max(o.spec, o.batch), err) ||
            (use_mtp &&
             !mtp.bind(last_st ? last_st->wt : wt, last_st ? &last_st->head : &native_head, ver.final_R_all(), err))) {
            std::fprintf(stderr, "strata serve: %s\n", err.c_str());
            return 1;
        }
        if (o.batch > 0) {   // the verifier gets the slot sessions; one that cannot take them runs without slots
            std::vector<strata::core::SessionState*> ptrs;
            ptrs.reserve(bslot_ss.size());
            for (auto& u : bslot_ss) ptrs.push_back(u.get());
            bool slots_ok = ver.init_slots(ptrs, err);
            // a layer split: every later stage's verifier takes its own GPU's sessions (--split-device 0: the same)
            for (int st = 1; slots_ok && st < n_stages; ++st) {
                std::vector<strata::core::SessionState*> sp_ptrs = ptrs;
                if (!split_same) {
                    sp_ptrs.clear();
                    for (auto& u : stages[(size_t) st - 1]->bslots) sp_ptrs.push_back(u.get());
                }
                slots_ok = stage_ver(st).init_slots(sp_ptrs, err);
            }
            if (!slots_ok) {
                std::fprintf(stderr, "strata serve: --batch is off: %s\n", err.c_str());
                err.clear();
                o.batch = 0;
            }
        }
        for (int st = 0; st < n_stages && n_stages > 1; ++st) {
            split_drive.plan[st] = stage_ver(st).plan_sink();
            if (st > 0) {
                stage_ver(st).set_split(o.spec_split);
                stage_ver(st).set_pcie_mode(o.pcie_mode == "dma" ? 0 : o.pcie_mode == "direct" ? 1 : 2);
            }
        }
        // the pool the verify windows call: with a layer split, the wrapper that routes each layer to its stage
        const strata::core::PoolMultiFn win_pool_fn = n_stages > 1 ? &drive_pool_split : &drive_pool_multi;
        void* const win_pool_user = n_stages > 1 ? (void*) &split_drive : (void*) &drive;
        mem_mark("the verifier and the drafter's binding");
        ver.set_split(o.spec_split);
        // auto: the copy kernel for every pack.  DMA (the native packs' default until 0.1.13) has the host call
        // cudaMemcpyAsync + cudaLaunchHostFunc inside a verify window while the GPU spins on the flag they raise;
        // issue #31's thread dumps show the host stuck in that cudaMemcpyAsync on a driver lock for good.  The copy
        // kernel needs no host CUDA call there, and costs ~1-3% decode on IQ3_S (45.3 -> 44.8 tok/s, 8 requests).
        ver.set_pcie_mode(o.pcie_mode == "dma" ? 0 : o.pcie_mode == "direct" ? 1 : 2);
        std::vector<int64_t> cur;
        // ---- the conversation cache (see ConvCheckpoint).  `live` is what the session holds right now: the tokens
        // it has consumed, so a request that starts with exactly them continues without any copy.  `checks` are the
        // saved points; every one of them is a prefix of `live` (the loop drops the rest), so they form a chain -
        // the radix cache's tree collapsed onto the one branch of history whose cells the session holds.  The
        // chain's root is the deepest point every request so far shared (the end of the system prompt, in
        // practice); the retention policy pins it and rotates the rest LRU (conv_cache.hpp), so a NEW chat that
        // shares that prefix mounts through it instead of reading it again.
        std::vector<int32_t> live;
        std::vector<ImgKey> live_imgs, req_imgs;
        bool live_ok = false;
        std::vector<ConvCheckpoint> checks;
        uint64_t check_clock = 0;   // the checkpoints' LRU clock; creation and every use advance it
        int64_t tail_ckpt_len = -1;   // --prompt-cache-tail: the length of the one tail checkpoint alive (-1 = none)
        bool cvec_cached = true;   // the control vector's state the live session and the checkpoints were read with
        // ---- parked conversations (--conversation-cache-mib, upstream ccc660b-cecf697): when a request continues
        // another conversation than the live one, the live one's K/V, running state and checkpoints are copied to
        // host RAM first, and a parked one whose tokens start the request is put back instead of read again.  Saved
        // only on a switch or rewind, not on each continuing request; no graph address changes (the images are
        // ordinary host vectors).
        strata::core::ConversationCache conversations(
            o.prompt_cache > 0 && use_mtp ? (size_t) o.conversation_cache_mib * 1024 * 1024 : 0,
            (size_t) o.conversation_cache_slots);
        // a layer split parks every GPU's session (its layers), each copied on its own device
        std::vector<strata::core::ConversationStage> conv_stages;
        if (multi_gpu) {
            conv_stages.push_back({&ss, 0});
            for (auto& st : stages) conv_stages.push_back({&st->ss, st->dev});
        }
        auto snapshot_bytes = [&](const strata::core::ConversationView& v, size_t& bytes, std::string& e) {
            return conv_stages.empty() ? strata::core::conversation_snapshot_bytes(v, ss, g, mtp.kv_state(), bytes, e)
                                       : strata::core::conversation_split_bytes(v, conv_stages, g, mtp.kv_state(),
                                                                                bytes, e);
        };
        auto snapshot_validate = [&](const strata::core::SavedConversation& image, std::string& e) {
            return conv_stages.empty() ? strata::core::conversation_snapshot_validate(image, ss, g, mtp.kv_state(), e)
                                       : strata::core::conversation_split_validate(image, conv_stages, g,
                                                                                   mtp.kv_state(), e);
        };
        // ---- --conversation-save: the parked conversations in a folder, read back after a restart.  A file is used
        // only by an engine with the same model files, engine version and settings its state depends on.
        strata::core::ConversationDisk disk;
        if (!o.conversation_save.empty()) {
            std::string fp = "engine=" STRATA_VERSION;
            auto identity = [&](const char* key, const std::string& path) {
                fp += std::string("\n") + key + "=" + strata::core::conversation_file_identity(path);
            };
            identity("pack", o.pack);
            identity("native", o.native_preset);
            identity("ple", o.ple_gguf);
            identity("embd", o.embd_gguf);
            identity("head", o.native_head_gguf);
            for (const auto& d : o.native_dense_gguf) identity("dense", d);
            identity("mtp", o.mtp);
            for (const auto& [path, scale] : o.cvec_files) {
                identity("cvec", path);
                fp += " scale=" + std::to_string(scale);
            }
            const char* rot = std::getenv("STRATA_KV_ROT");
            fp += "\nkv=" + o.kv + " rot=" + (rot ? rot : "") + " max_context=" + std::to_string(o.max_context) +
                  " mtp_window=" + std::to_string(o.mtp_window) + " turn_token=" + std::to_string(o.turn_token) +
                  " no_ple=" + std::to_string((int) o.no_ple) + " cvec=" + cvec_summary;
            if (multi_gpu) {   // a layer split's files hold each GPU's state: read only by the same split
                fp += "\nsplit=";
                for (const int64_t k : split_at) fp += std::to_string(k) + ",";
            }
            strata::core::ConversationDiskOptions dopt;
            dopt.dir = o.conversation_save;
            dopt.fingerprint = std::move(fp);
            dopt.budget = (size_t) o.conversation_save_mib * 1024 * 1024;
            dopt.max_age = std::chrono::hours(o.conversation_save_hours);
            dopt.checkpoints = (size_t) o.conversation_save_checkpoints;
            dopt.compress = o.conversation_save_compress;
            dopt.stage_parts = multi_gpu ? stages.size() : 0;
            std::string de;
            if (disk.open(std::move(dopt), de))
                std::fprintf(stderr, "strata serve: conversation save: %s holds %zu usable conversations (%zu MiB in "
                                     "the folder)%s\n", o.conversation_save.c_str(), disk.size(),
                             disk.folder_bytes() >> 20, o.conversation_save_compress ? "; writing compressed" : "");
            else
                std::fprintf(stderr, "strata serve: conversation save: off (%s)\n", de.c_str());
        }
        Clock::time_point disk_expired_at = Clock::now();
        // a failed write loses only the saved copy: the conversation is read again when it comes back
        auto save_one = [&](const strata::core::SavedConversation& image, const char* why) {
            if (!disk.enabled()) return;
            const auto t0 = Clock::now();
            size_t written = 0;
            std::string de;
            if (disk.save(image, written, de))
                std::fprintf(stderr, "strata serve: conversation save: wrote %zu tokens (%s, %zu MiB) in %.1f ms; "
                                     "files=%zu folder=%zu MiB\n", image.live.ids.size(), why, written >> 20,
                             std::chrono::duration<double, std::milli>(Clock::now() - t0).count(), disk.size(),
                             disk.folder_bytes() >> 20);
            else
                std::fprintf(stderr, "strata serve: conversation save: %zu tokens not written (%s)\n",
                             image.live.ids.size(), de.c_str());
        };
        std::vector<strata::core::SavedConversation> evicted;   // what the RAM cache pushed out, until written
        auto save_evicted = [&]() {
            for (const auto& image : evicted) save_one(image, "evicted");
            evicted.clear();
        };
        // Disk sessions: what a session file is bound to.  The model fingerprint samples every model input this
        // engine loaded, by role (conversation_file.hpp), once; the config fingerprint covers the RESOLVED settings
        // that change what the saved bytes mean - the rope (K is cached post-RoPE), the loaded control vector, the
        // K/V format and the arithmetic switches.  Sampling, seeds, draft tuning and the expert tier are not in it.
        std::optional<uint64_t> model_fp, config_fp;
        auto session_identity = [&](strata::core::SessionFileIdentity& id, std::string& e,
                                    const std::function<void()>& next_file) -> bool {
            if (!model_fp) {
                // what the loaders actually resolved: the CLI inputs, the pack's files and - from the expert source
                // itself - every file its experts were read from (native_experts.txt can name GGUFs per layer/role)
                strata::core::SessionInputs in;
                if (!o.native_preset.empty()) in.native_shards = strata::gguf_split_paths(o.native_preset);
                in.native_dense = o.native_dense_gguf;
                if (!o.native_head_gguf.empty())
                    in.native_head = o.native_head_gguf == o.native_preset ? in.native_shards
                                                                           : strata::gguf_split_paths(o.native_head_gguf);
                in.embedding = o.embd_gguf;
                in.ple = o.ple_gguf;
                in.pack = o.pack;
                in.mtp = o.mtp.empty() ? std::string() : o.mtp + "/dense.bin";
                if (srcp != nullptr) in.experts = srcp->model_inputs();
                std::vector<strata::core::SessionModelFile> files = strata::core::session_model_inputs(in);
                // the pack's other files the loader reads (weights.cpp, expert_layout.cpp)
                if (!o.pack.empty()) {
                    for (const char* n : {"index.txt", "dense.bin"}) files.push_back({std::string("pack ") + n, o.pack + "/" + n});
                    for (const char* n : {"embd.bin", "extra.bin", "manifest.json"})
                        files.push_back({std::string("pack ") + n, o.pack + "/" + n, true});
                    if (srcp == nullptr) files.push_back({"pack experts.bin", o.pack + "/experts.bin", true});
                }
                if (!o.mtp.empty()) {
                    for (const char* n : {"dense.txt", "experts.bin"}) files.push_back({std::string("mtp ") + n, o.mtp + "/" + n});
                    files.push_back({"mtp draft_vocab.bin", o.mtp + "/draft_vocab.bin", true});
                }
                uint64_t fp = 0;
                // each file is one blocking read of at most 2 MiB (head and tail): announced before it starts
                if (next_file) next_file();
                if (!strata::core::session_model_fingerprint(files, fp, e, next_file))
                    return false;
                model_fp = fp;
            }
            if (!config_fp) {
                strata::core::SessionConfig c;
                c.engine_version = STRATA_VERSION;
#if defined(STRATA_USE_HIP)
                c.backend = "hip";
#else
                c.backend = "cuda";
#endif
                c.kv = o.kv;
                c.max_context = o.max_context;
                c.kv_resident = o.kv_resident;
                c.mtp_window = o.mtp.empty() ? -1 : o.mtp_window;
                const char* rot = std::getenv("STRATA_KV_ROT");
                c.kv_rot = rot != nullptr && rot[0] == '1';
                c.rope = {(int64_t) rope_cfg.type, rope_cfg.freq_base, rope_cfg.factor, rope_cfg.freq_scale(),
                          rope_cfg.orig_ctx, rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast,
                          rope_cfg.beta_slow};
                c.cvec = cvec_digest;
                c.switches = {
                    {"no_ple", o.no_ple}, {"native_bf16", o.native_bf16}, {"native_bf16_extra", o.native_bf16_extra},
                    {"native_ple_key", o.native_ple_key}, {"native_moe_combine", o.native_moe_combine},
                    {"native_gdn", o.native_gdn}, {"native_flash_attn_short", o.native_flash_attn_short},
                    {"native_qsa_indexer", o.native_qsa_indexer}, {"native_qsa", o.native_qsa},
                    {"native_rope", o.native_rope}, {"native_ple_postops", o.native_ple_postops},
                    {"native_router", o.native_router}, {"cpu_oracle_q8_0", o.cpu_oracle_q8_0},
                    {"gr_fp32_activations", o.gr_fp32_activations}, {"gr_native_mmvf", o.gr_native_mmvf},
                    {"native_preset", !o.native_preset.empty()}, {"shared_late", o.shared_late},
                    {"keep_canonical", o.keep_canonical}, {"no_fused_gr", o.no_fused_gr},
                    {"no_fast_attn", o.no_fast_attn}, {"no_fused_gdn", o.no_fused_gdn},
                    {"no_fast_select", o.no_fast_select}, {"vision", o.vision}};
                config_fp = strata::core::session_config_fingerprint(c);
            }
            id.model = *model_fp;
            id.config = *config_fp;
            return true;
        };
        auto park_current = [&](size_t held) -> bool {
            if (!conversations.enabled() || !live_ok || live.empty()) return true;
            const strata::core::ConversationView view{live, live_imgs, checks, cvec_cached};
            auto reuse = conversations.take_reuse();
            size_t estimate = 0;
            if (!snapshot_bytes(view, estimate, err)) {
                std::fprintf(stderr, "strata serve: conversation cache: skip parking (%s)\n", err.c_str());
                err.clear();   // a recoverable miss must not poison the batched draft prefill's error channel
                return true;
            }
            const size_t fresh_estimate = estimate;
            if (!reuse.kv.empty() && !strata::core::conversation_snapshot_capture_bytes(
                    reuse, view, ss, g, mtp.kv_state(), estimate, err)) {
                reuse = {};
                estimate = fresh_estimate;
                err.clear();
            }
            // A park carrying its own retained K/V replaces memory the cache already held, so capacity is
            // make_room's call either way and put()'s accounting still bounds the budget.  The with-reuse estimate
            // stays uncapped: it counts the retained buffers' capacity and directories, which put() charges too.
            // #342: before make_room evicts oldest-first, the copies of this conversation a turn back go (they hold
            // nothing the outgoing chain does not, apart from the tail this conversation rewrote)
            if (const size_t dropped = conversations.drop_superseded(live, live_imgs, checks, cvec_cached))
                std::fprintf(stderr, "strata serve: conversation cache: dropped %zu superseded cop%s of this "
                             "conversation; parked=%zu\n", dropped, dropped == 1 ? "y" : "ies", conversations.size());
            const bool room = conversations.make_room(estimate, held, disk.enabled() ? &evicted : nullptr);
            save_evicted();   // before the capture allocates: the pushed-out images are freed once written
            if (!room) {
                std::fprintf(stderr, "strata serve: conversation cache: skip parking (snapshot %zu MiB exceeds available "
                                     "budget)\n", estimate >> 20);
                return true;
            }
            const auto t0 = Clock::now();
            try {
                const uint64_t floor = (uint64_t) o.conversation_cache_min_free_mib * 1024 * 1024;
                const size_t additional = estimate - reuse.bytes();
                if (!strata::core::conversation_memory_admit(strata::core::conversation_available_memory(),
                                                             additional, floor)) {
                    std::fprintf(stderr, "strata serve: conversation cache: skip parking (physical RAM admission; need "
                                         "%zu MiB plus %lld MiB floor, or telemetry unavailable)\n",
                                 additional >> 20, (long long) o.conversation_cache_min_free_mib);
                    return true;
                }
                strata::core::SavedConversation image;
                size_t reused_bytes = 0;
                if (conv_stages.empty() ? !strata::core::conversation_snapshot_save(image, view, ss, g, mtp.kv_state(),
                                                                                    err, std::move(reuse), &reused_bytes)
                                        : !strata::core::conversation_split_save(image, view, conv_stages, g,
                                                                                 mtp.kv_state(), err))
                    return false;
                if (!strata::core::conversation_memory_admit(strata::core::conversation_available_memory(), 0, floor)) {
                    std::fprintf(stderr, "strata serve: conversation cache: skip parking (physical RAM floor after "
                                         "capture, or telemetry unavailable)\n");
                    return true;
                }
                const size_t snapshot_bytes = image.bytes();
                const bool stored = conversations.put(std::move(image), held, disk.enabled() ? &evicted : nullptr);
                std::fprintf(stderr, "strata serve: conversation cache: %s %zu tokens in %.1f ms; parked=%zu bytes=%zu "
                                     "evictions=%zu snapshot_bytes=%zu reused_kv_bytes=%zu\n",
                             stored ? "parked" : "skipped", live.size(),
                             std::chrono::duration<double, std::milli>(Clock::now() - t0).count(),
                             conversations.size(), conversations.bytes(), conversations.evictions(), snapshot_bytes,
                             reused_bytes);
                save_evicted();
            } catch (const std::bad_alloc&) {
                evicted.clear();
                // the live session is untouched: read the prompt the ordinary way instead of ending the server
                std::fprintf(stderr, "strata serve: conversation cache: allocation failed; skip parking\n");
            }
            return true;
        };
        int64_t pp_total = 0, pp_from = 0, pp_next_check = 0;
        bool pp_tail_saved = false;
        // #471: the position the prompt pass has read up to (a chunk's or a window's end): what a request cancelled
        // mid-read reports as read, instead of the whole prompt
        int64_t pp_reached = 0;
        Clock::time_point pp_t0 = Clock::now();
        auto imgs_below = [&](const std::vector<ImgKey>& all, int64_t L) {
            std::vector<ImgKey> v;
            for (const ImgKey& k : all) if (k.start < L) v.push_back(k);
            return v;
        };
        // a checkpoint of the state after `cur[0, L)`; false only when the copy itself failed
        // A layer split's mid-prompt checkpoints: when the last stage reports a chunk, the earlier ones already read
        // the next, so each stage saves its own part when IT reaches a checkpoint position (the same rule as below:
        // every `prompt_cache_every` tokens from where the request resumed), and the last stage puts them together.
        std::mutex part_mu;
        std::map<int64_t, std::vector<ConvCheckpoint>> part_at;   // position -> one part per stage
        std::vector<int64_t> part_next(stages.size() + 1, INT64_MAX);
        // a checkpoint of the state after `cur[0, L)`; false only when the copy itself failed.  `parts`: the stages'
        // states saved at L (a split's mid-prompt checkpoint); without, they are read now (everything is at L)
        auto checkpoint_at = [&](int64_t L, std::vector<ConvCheckpoint>* parts = nullptr, bool as_tail = false) -> bool {
            if (o.prompt_cache <= 0 || L < 1) return true;
            for (ConvCheckpoint& c : checks)
                if ((int64_t) c.ids.size() == L) {
                    c.used = ++check_clock;
                    if (!as_tail && L == tail_ckpt_len) tail_ckpt_len = -1;   // a periodic one lands here: a normal item now
                    return true;
                }
            ConvCheckpoint c;
            c.ids.assign(cur.begin(), cur.begin() + L);
            c.imgs = imgs_below(req_imgs, L);
            if (parts != nullptr) {
                if (parts->size() != stages.size() + 1) return false;
                c.gdn = std::move((*parts)[0].gdn);
                c.ple = std::move((*parts)[0].ple);
                c.tails = std::move((*parts)[0].tails);
                c.dead = std::move((*parts)[0].dead);
                c.block_pos = std::move((*parts)[0].block_pos);
                for (size_t i = 1; i < parts->size(); ++i) c.stage_parts.push_back(std::move((*parts)[i]));
            } else {
                if (!strata::gpu::device_sync() || !checkpoint_save(c, ss, g)) return false;
                for (auto& st : stages) {   // a layer split's later stages: their sessions' part
                    const strata::core::OnDevice on(st->dev);
                    ConvCheckpoint part;
                    part.ids = c.ids;
                    if (!strata::gpu::device_sync() || !checkpoint_save(part, st->ss, g)) return false;
                    c.stage_parts.push_back(std::move(part));
                }
            }
            c.used = ++check_clock;
            if (as_tail) {   // only one tail checkpoint stays alive: the previous request's goes
                const int64_t old_tail = tail_ckpt_len;
                checks.erase(std::remove_if(checks.begin(), checks.end(), [&](const ConvCheckpoint& k) {
                                 return old_tail >= 0 && (int64_t) k.ids.size() == old_tail;
                             }), checks.end());
                tail_ckpt_len = L;
            } else if (L == tail_ckpt_len) {
                tail_ckpt_len = -1;   // the flagged tail was dropped and a normal checkpoint takes its length
            }
            checks.push_back(std::move(c));
            while ((int) checks.size() > o.prompt_cache) {
                std::vector<uint64_t> stamps;
                std::vector<char> is_tail;
                stamps.reserve(checks.size());
                for (const ConvCheckpoint& k : checks) {
                    stamps.push_back(k.used);
                    is_tail.push_back((char) (tail_ckpt_len >= 0 && (int64_t) k.ids.size() == tail_ckpt_len));
                }
                // the tail checkpoint serves only a branch of the last request: it leaves before a periodic one
                std::unique_ptr<bool[]> tail_flags(new bool[is_tail.size()]);
                for (size_t i = 0; i < is_tail.size(); ++i) tail_flags[i] = is_tail[i] != 0;
                const size_t victim = strata::program::conv_cache::eviction_victim(
                    stamps.data(), stamps.size(), o.prompt_cache, tail_flags.get());
                checks.erase(checks.begin() + (std::ptrdiff_t) victim);
            }
            return true;
        };
        sp.on_chunk = [&](const float* R_rows, int64_t T, int64_t p0, std::string& e) -> bool {
            std::vector<int32_t> nxt((size_t) T);
            for (int64_t t = 0; t < T; ++t) nxt[(size_t) t] = (int32_t) cur[(size_t) (p0 + t + 1)];
            // E-9: batched through the prompt path when it can (one GPU: a layer split's drafter is on the last stage)
            const bool batched = use_mtp && !multi_gpu && sp.draft_kv(mtp, R_rows, nxt.data(), T, p0, e);
            if (!e.empty() || (use_mtp && !batched && !mtp.prefill(R_rows, nxt.data(), T, p0, e))) return false;
            // progress for the server window: PP <position reached> <prompt tokens> <ms> <fresh tokens/s>
            const int64_t done = p0 + T;
            pp_reached = done;
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - pp_t0).count();
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) pp_total, ms,
                        ms > 0.0 ? 1000.0 * (double) (done - pp_from) / ms : 0.0);
            strata::core::progress_at("reading the prompt (batched), done up to token", done);
            strata::core::progress_beat();
            std::fflush(stdout);
            const bool periodic_checkpoint = o.prompt_cache_every > 0 && done >= pp_next_check;
            // Keep the prefill's existing chunk geometry. One extra near-tail checkpoint
            // can serve a branch whose shared prefix ends before the final cached state.
            const bool tail_checkpoint = o.prompt_cache_tail && o.prompt_cache > 0 && o.prompt_cache_every > 0 &&
                !multi_gpu && !pp_tail_saved && pp_total - done > std::max<int64_t>(1, o.short_read) &&
                pp_total - done <= T;   // this chunk's real size (--prefill auto chunks differ from o.prefill_chunk)
            if (periodic_checkpoint || tail_checkpoint) {
                bool saved = false;
                if (multi_gpu) {   // the stages' parts, saved when each of them read this chunk
                    std::vector<ConvCheckpoint> parts;
                    {
                        std::lock_guard<std::mutex> lk(part_mu);
                        auto it = part_at.find(done);
                        if (it != part_at.end()) parts = std::move(it->second);
                        part_at.erase(part_at.begin(), part_at.upper_bound(done));
                    }
                    // a part a stage saved has its tokens (its running state can be empty: a stage of QSA layers only,
                    // as the RTX 4070's layer 47 with K=47 - which dropped every mid-prompt checkpoint of a split)
                    bool complete = parts.size() == stages.size() + 1;
                    for (const ConvCheckpoint& k : parts) complete = complete && !k.ids.empty();
                    saved = !complete || checkpoint_at(done, &parts);   // an incomplete set: no checkpoint here
                } else {
                    saved = checkpoint_at(done, nullptr, tail_checkpoint && !periodic_checkpoint);
                }
                if (!saved) { e = "saving a conversation checkpoint failed"; return false; }
                if (periodic_checkpoint) pp_next_check = done + o.prompt_cache_every;
                if (tail_checkpoint) pp_tail_saved = true;
            }
            return true;
        };
        if (multi_gpu) {   // the batched prompt is reported by its last stage (the drafter's rows are there)
            stages.back()->sp.on_chunk = std::move(sp.on_chunk);
            sp.on_chunk = nullptr;
            for (size_t i = 0; i <= stages.size(); ++i) {
                strata::prefill::Prefill& stage_sp = i == 0 ? sp : stages[i - 1]->sp;
                strata::core::SessionState& stage_ss = i == 0 ? ss : stages[i - 1]->ss;
                stage_sp.on_stage_chunk = [&, i](int64_t done, std::string& e) -> bool {
                    if (o.prompt_cache <= 0 || o.prompt_cache_every <= 0 || done < part_next[i]) return true;
                    part_next[i] = done + o.prompt_cache_every;
                    ConvCheckpoint part;   // this stage's state at `done` (its stream is synchronized)
                    part.ids.assign(cur.begin(), cur.begin() + done);
                    if (!checkpoint_save(part, stage_ss, g)) { e = "saving a checkpoint part failed"; return false; }
                    std::lock_guard<std::mutex> lk(part_mu);
                    auto& v = part_at[done];
                    v.resize(stages.size() + 1);
                    v[i] = std::move(part);
                    return true;
                };
            }
        }
        drive.d.plan = ver.plan_sink();
        drive.d.pcie_num = std::max(0, std::min(256, (int) (o.pcie_frac * 256.0 + 0.5)));
        if (o.adapt_every > 0 && o.adapt_swaps > 0) drive.d.usage.assign((size_t) (g.n_layers * g.n_expert), 0.0f);
        // #477 --expert-profile-save: what the adaptive tier learned, kept across restarts (opt-in; off: `heat` stays
        // empty and nothing below runs).  It needs the adaptive tier's counts and the residency table.
        std::vector<double> heat;
        if (!o.expert_profile_save.empty()) {
            if (drive.d.usage.empty() || host_res.empty())
                std::fprintf(stderr, "strata serve: --expert-profile-save needs the adaptive tier (--adapt-every and "
                                     "--adapt-swaps above 0) and --expert-profile: nothing will be saved\n");
            else
                heat.assign(drive.d.usage.size(), 0.0);
        }
        Clock::time_point profile_saved_at = Clock::now();
        strata::gpu::Stream adapt_stream = nullptr;
        if (!(adapt_stream = strata::gpu::stream_create())) {
            std::fprintf(stderr, "strata serve: cannot create the refill stream\n");
            return 1;
        }
        // plan v0.3 P6: swaps in flight - (residency index, slot) admitted when adapt_ev has completed
        std::vector<std::pair<int32_t, int32_t>> pending;
        strata::gpu::Event* adapt_ev = nullptr;
        strata::gpu::event_create(&adapt_ev);
        // a layer split's later stages keep a copy of the residency table on their devices, and swap on their own
        auto res_upload = [&]() {
            if (d_res != nullptr)
                strata::gpu::copy(d_res, host_res.data(), host_res.size() * sizeof(int32_t));
            for (auto& st : stages) {
                const strata::core::OnDevice on(st->dev);
                strata::gpu::copy(st->d_res, host_res.data(), host_res.size() * sizeof(int32_t));
            }
        };
        auto apply_pending = [&](bool wait) {
            if (pending.empty()) return;
            if (wait) strata::gpu::event_sync(adapt_ev);
            else if (!strata::gpu::event_query(adapt_ev)) return;
            for (auto& st : stages)
                if (st->adapt_live) {
                    if (wait) strata::gpu::event_sync(st->adapt_ev);
                    else if (!strata::gpu::event_query(st->adapt_ev)) return;
                }
            for (auto& st : stages) st->adapt_live = false;
            for (const auto& [i, slot] : pending) {
                host_res[(size_t) i] = slot;
                srcp->release((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);   // in VRAM now: RAM not needed
            }
            src.commit_exchanges();
            pending.clear();
            res_upload();
        };
        // the VRAM tier follows the conversation (the same rule as the speculative loop below)
        int64_t adapt_swaps_n = 0, adapt_sync_n = 0;   // STRATA_DECODE_TIMING: swaps issued, of them synchronous
        auto adapt = [&]() -> bool {
            if (!pending.empty()) return true;   // the previous swaps are still in flight
            struct Swap { float gain; int32_t layer, in, out; };
            std::vector<Swap> swaps;
            std::vector<std::pair<float, int32_t>> cand, vict;
            for (int64_t l = 0; l < g.n_layers; ++l) {
                cand.clear();
                vict.clear();
                const float* u = drive.d.usage.data() + l * g.n_expert;
                const int32_t* r = host_res.data() + l * g.n_expert;
                for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                    if (r[e] < 0) { if (u[e] >= 2.0f && !helper_holds(drive.d, l, e)) cand.emplace_back(u[e], e); }
                    else vict.emplace_back(u[e], e);
                }
                if (cand.empty() || vict.empty()) continue;
                std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
                const size_t nc = std::min(cand.size(), vict.size());
                std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                                  [](auto& a, auto& b) { return a.first < b.first; });
                for (size_t i = 0; i < nc; ++i) {
                    if (cand[i].first < vict[i].first + 1.5f) break;
                    swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
                }
            }
            std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
            if ((int) swaps.size() > o.adapt_swaps) swaps.resize((size_t) o.adapt_swaps);
            bool main_live = false;
            auto swap_home = [&](int32_t layer) {   // the cache that owns the layer, on its card's stream
                const int stn = multi_gpu ? stage_of(layer) : 0;
                GpuStage* gs = stn > 0 ? stages[(size_t) stn - 1].get() : nullptr;
                return SwapHome{gs ? &gs->cache : &xcache, gs ? gs->adapt_stream : adapt_stream, gs ? gs->dev : -1};
            };
            if (!resident_stage_swaps(src, host_res, g.n_expert, swaps, swap_home)) return false;
            for (const Swap& s : swaps) {
                const size_t in = (size_t) s.layer * g.n_expert + s.in, out = (size_t) s.layer * g.n_expert + s.out;
                const int32_t slot = host_res[out];
                // a transient source's blob (the GGUF read in place) is copied into fill_tmp: that swap is synchronous
                const bool tb = srcp->transient(s.layer, s.in);
                const uint8_t* b = blob_to_copy(srcp, s.layer, s.in, fill_tmp);
                const int stn = multi_gpu ? stage_of(s.layer) : 0;   // the swap stays in the layer's own cache
                GpuStage* gs = stn > 0 ? stages[(size_t) stn - 1].get() : nullptr;
                const strata::core::OnDevice on(gs ? gs->dev : -1);
                uint8_t* sdst = gs ? gs->cache.device_slot(slot) : xcache.device_slot(slot);
                const size_t sn = (size_t) strata::kernels::cpu::expert_layout().blob_bytes(s.layer);
                ++adapt_swaps_n;
                if (tb) ++adapt_sync_n;
                if (slot < 0 || b == nullptr ||
                    !(tb ? strata::gpu::copy(sdst, b, sn)
                         : strata::gpu::copy_async(sdst, b, sn, gs ? gs->adapt_stream : adapt_stream)))
                    return false;
                if (gs) gs->adapt_live = true;
                else main_live = true;
                host_res[out] = strata::core::kNotResident;   // evicted now: the CPU computes it meanwhile
                srcp->prefetch(s.layer, s.out);   // a file-backed arena released its pages: read them back ahead
                pending.emplace_back((int32_t) in, slot);      // resident once the copy has landed
            }
            if (!swaps.empty()) strata::gpu::event_record(adapt_ev, adapt_stream);
            (void) main_live;
            for (auto& st : stages)
                if (st->adapt_live) {
                    const strata::core::OnDevice on(st->dev);
                    strata::gpu::event_record(st->adapt_ev, st->adapt_stream);
                }
            // #477: the routing counted since the start (each count adds up to 1 / (1 - --adapt-decay) over its
            // decays: the sum is proportional to the routing itself) - only with --expert-profile-save, else `heat`
            // is empty
            for (size_t i = 0; i < heat.size(); ++i) heat[i] += (double) drive.d.usage[i];
            for (float& v : drive.d.usage) v *= o.adapt_decay;
            return true;
        };
        // #477: write the learned profile (between requests and at QUIT: a prompt's lent slots are back by then).
        // A swap still in flight counts as done - its expert is resident once the copy lands.  `why`: for the log.
        auto save_profile = [&](const char* why) {
            if (heat.empty()) return;
            std::vector<uint8_t> resident(host_res.size(), 0);
            for (size_t i = 0; i < host_res.size(); ++i) resident[i] = host_res[i] >= 0;
            for (const auto& p : pending) resident[(size_t) p.first] = 1;
            std::string e;
            const auto ranked = strata::core::rank_learned_profile(g.n_layers, g.n_expert, resident, heat,
                                                                   profile_loaded);
            if (strata::core::write_expert_profile(o.expert_profile_save, g.n_layers, g.n_expert, ranked, e))
                std::fprintf(stderr, "strata serve: expert profile saved to %s (%s)\n", o.expert_profile_save.c_str(),
                             why);
            else
                std::fprintf(stderr, "strata serve: the expert profile was not saved: %s\n", e.c_str());
            profile_saved_at = Clock::now();
        };
        // stdin is read on its own thread, so a STOP line reaches a request that is still running (the client went
        // away, or pressed Esc): the flag is checked between prompt chunks and between verify windows.
        std::atomic<bool> stop_req{false};
        std::mutex in_mu;
        std::condition_variable in_cv;
        std::deque<std::string> in_lines;
        bool in_eof = false;
        std::thread([&] {
            // read(2) on the descriptor, not std::cin: glibc's exit() flushes every stdio stream and waits for
            // stdin's lock, which getline holds while it waits for input - an engine ending on an error (every
            // std::exit) would hang in exit() on Linux, and the server would wait for it forever
            std::string l, buf;
            char chunk[4096];
            auto getline_fd = [&](std::string& out) -> bool {
                for (;;) {
                    const size_t nlpos = buf.find('\n');
                    if (nlpos != std::string::npos) {
                        out.assign(buf, 0, nlpos);
                        buf.erase(0, nlpos + 1);
                        return true;
                    }
                    const ssize_t n = ::read(0, chunk, sizeof chunk);
                    if (n < 0 && errno == EINTR) continue;
                    if (n <= 0) {
                        if (buf.empty()) return false;
                        out.swap(buf);
                        buf.clear();
                        return true;
                    }
                    buf.append(chunk, (size_t) n);
                }
            };
            while (getline_fd(l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                if (l == "STOP") { stop_req.store(true); continue; }
                std::lock_guard<std::mutex> lk(in_mu);
                in_lines.push_back(l);
                in_cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(in_mu);
            in_eof = true;
            in_cv.notify_one();
        }).detach();
        auto next_line = [&](std::string& out) -> bool {
            std::unique_lock<std::mutex> lk(in_mu);
            in_cv.wait(lk, [&] { return !in_lines.empty() || in_eof; });
            if (in_lines.empty()) return false;
            out = std::move(in_lines.front());
            in_lines.pop_front();
            return true;
        };
        sp.should_stop = [&] { return stop_req.load(); };
        // STRATA_TRACE=1: one stderr line per step of a request (the log shows where a request stops)
        const bool trace = std::getenv("STRATA_TRACE") != nullptr;
        auto tr = [&](const char* what, long long a = -1, long long b = -1) {
            if (!trace) return;
            std::fprintf(stderr, "strata trace: %s %lld %lld\n", what, a, b);
            std::fflush(stderr);
        };
        {
            // Check the reserve once everything is allocated.
            size_t free_b = 0, total_b = 0;
            strata::gpu::mem_info(&free_b, &total_b);
            // below ~256 MiB a later allocation (a first-used window's buffers, the desktop, another program) can make
            // the driver page GPU memory, and a verify graph spinning on a host flag then never finishes
            const int64_t free_mib = (int64_t) (free_b >> 20);
            if (free_mib >= 256) {
                std::fprintf(stderr, "strata serve: %lld MiB of VRAM free with everything loaded\n", (long long) free_mib);
            } else if (reserve_adapted) {
                // #496: the reserve was already lowered to make the cache fit - a bigger one would leave it no room
                std::fprintf(stderr, "strata serve: WARNING: %lld MiB of VRAM free with everything loaded - this card "
                                     "only just fits the model (the VRAM reserve was lowered to %d MiB so the expert "
                                     "cache fits): requests may stall. Close other programs that use the GPU, or lower "
                                     "--max-context\n", (long long) free_mib, o.vram_reserve_mib);
            } else {
                std::fprintf(stderr, "strata serve: %lld MiB of VRAM free with everything loaded - LOW: requests may stall;"
                                     " add --vram-reserve-mib %lld to the config's args (or lower --max-context)\n",
                             (long long) free_mib, (long long) (o.vram_reserve_mib + 512 - free_mib));
                // #781 / #831: an explicit --expert-cache N is a byte budget that is not checked against the VRAM once
                // the slots are written (auto is), so on a card it fills a smaller N is the cure: a few hundred slots
                // can be the difference between 14 and 100 tok/s under WDDM.  Said, never changed.
                if (!auto_cache)
                    std::fprintf(stderr, "strata serve: the expert cache was set with --expert-cache (%lld slots): a "
                                         "smaller one - or --expert-cache auto, which checks the free VRAM once the "
                                         "slots are written - leaves room; decode can be several times slower with the "
                                         "card this full\n", (long long) xcache.slots());
            }
        }
        // what the server's Monitor tab shows (servers before 0.1.8 skip unknown lines until READY)
        {
            size_t free_b = 0, total_b = 0;
            strata::gpu::mem_info(&free_b, &total_b);
            std::printf("INFO context=%lld kv=%s kv_resident=%lld expert_slots=%lld expert_cache_mib=%lld spec=%d "
                        "mtp_max=%d lookup=%d vram_free_mib=%lld cvec=%s arena_mib=%lld pool_workers=%d pcie_frac=%.2f "
                        "spec_min_p=%.2f conversation_cache_mib=%lld conversation_cache_slots=%d "
                        "conversation_cache_min_free_mib=%lld tail_role_token=%lld vram_elastic=%d batch_slots=%d slot_cache=%d "
                        "engine=" STRATA_VERSION "\n",
                        (long long) o.max_context, o.kv.c_str(),
                        (long long) (g.n_qsa_layers() > 0 && ss.qsa_states[ss.qsa_primary()].kv_mode == 1
                                         ? ss.qsa_states[ss.qsa_primary()].n_slots * 4 : 0),
                        (long long) xcache.slots(), (long long) (xcache.bytes() >> 20), o.spec, o.mtp_max_t,
                        o.suffix_draft, (long long) (free_b >> 20), cvec_summary.c_str(),
                        (long long) (strata::kernels::cpu::expert_layout().total >> 20), pool.workers(), o.pcie_frac,
                        o.spec_min_p, (long long) o.conversation_cache_mib, o.conversation_cache_slots,
                        (long long) o.conversation_cache_min_free_mib, (long long) o.tail_role_token,
                        xcache.segmented() ? 1 : 0, o.batch, o.batch > 0 && o.prompt_cache > 0 ? 1 : 0);
        }
        // issue #29: a request whose heartbeat (tokens, prompt chunks, verify windows) stops for this long is stuck on
        // a flag nobody will raise - end the engine with where it was, so the server starts it again instead of the
        // GPU spinning forever.  STRATA_WATCHDOG_S=0 turns it off.  Issue #31: before it does, it reports what every
        // part was doing (stall_report), so one occurrence says where the wait is.
        {
            const char* ws = std::getenv("STRATA_WATCHDOG_S");
            const int limit = ws ? std::atoi(ws) : 60;   // one step (a prompt layer, a verify window) takes seconds
            if (limit > 0)
                std::thread([limit] {
                    strata::core::Progress& p = strata::core::progress();
                    uint64_t last = p.beats.load(), ticks_at = p.ticks.load();
                    auto since = std::chrono::steady_clock::now();
                    for (;;) {
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                        const auto now = std::chrono::steady_clock::now();
                        const uint64_t b = p.beats.load();
                        if (!p.busy.load() || b != last) { last = b; ticks_at = p.ticks.load(); since = now; continue; }
                        if (now - since < std::chrono::seconds(limit)) continue;
                        // a blocking step's explicit allowance (session files): still within it, not yet stuck
                        if (strata::core::progress_now_ms() < p.allow_until_ms.load()) continue;
                        std::fprintf(stderr, "strata serve: no progress for %d s during a request (%s) - stopping "
                                             "the engine so the server starts it again (issue #29)\n",
                                     limit, stage_text().c_str());
                        stall_report(stderr, p.ticks.load() - ticks_at);
                        std::fflush(stderr);
                        std::abort();
                    }
                }).detach();
        }
        std::printf("READY %lld stop\n", (long long) o.max_context);   // "stop": this engine honours STOP
        std::fflush(stdout);
        std::string line;
        int64_t rounds = 0;
        const int S = o.spec;
        const int S_mtp = o.mtp_max_t > 0 ? std::min(o.mtp_max_t, S) : S;   // the MTP's windows; suffixes go up to S
        if (use_mtp && S_mtp < S) mtp.set_max_drafts(S_mtp - 1);
        strata::spec::SuffixDrafter sfx(std::max(1, o.suffix_draft), 64, (size_t) o.max_context + 4096);
        strata::spec::DraftPolicy policy(S);   // MTP or lookup window, learned over the whole process
        // The vision path (--vision): GENI <max_new> <embeddings file> <id,id,...> carries images.  The file is one
        // or more strata-vision records (int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats) in prompt order;
        // each image's rows go to its run of <|image_pad|> tokens, whose M-RoPE positions are mtmd's: t = p,
        // h = p + y, w = p + x, and the text after the image continues at p + max(nx, ny).
        constexpr int64_t kImagePad = 248056;   // qwen4exp.ple.image_token_id: the PLE hash reads it for image cells
        bool mrope_identity = true;
        std::vector<float> img_rows;
        std::vector<const float*> row_ptr;
        // ---- #533: VRAM <reserve_mib>, between requests, only with --vram-elastic.  It shrinks the expert cache
        // until that much VRAM is free for other programs, or grows it back towards its full size when more than that
        // is free.  Never on its own: only this command (the server's POST /v1/vram) moves it.  A shrink gives back
        // the cache's LAST segments: the experts in their slots become CPU misses, as any expert outside the cache is,
        // and the prompt path's loan moves down to the end of what is left (a smaller chunk when it no longer fits).
        // A grow maps them again and puts each slot's expert back (or, when the adaptive tier already brought that
        // one back, the most-routed missing expert of the same layer).
        std::vector<std::pair<int32_t, int32_t>> vram_evicted;   // (residency index, slot) the shrinks took
        const int64_t chunk_full = o.prefill_chunk;              // the loan's chunk with the whole cache
        auto relend = [&]() {   // the prompt path's loan: the end of the slots left, its chunk as large as fits
            if (lend_first < 0) return;
            const int64_t live = xcache.slots();
            auto fits = [&](int64_t c) { const int64_t k = lend_slots(c); return k > 0 && k + 128 <= live; };
            int64_t c = chunk_full;
            while (c > 256 && !fits(c)) c = std::max<int64_t>(256, c / 2 / 256 * 256);
            o.prefill_chunk = c;
            lend_first = (int32_t) std::max<int64_t>(0, live - lend_slots(c));   // laid out again at the next loan
        };
        auto vram_command = [&](const std::string& cmd, std::string& e) -> bool {
            // `VRAM` alone: the reserve the engine started with (--vram-reserve-mib)
            char* end = nullptr;
            const bool bare = cmd.find_first_not_of(' ', 4) == std::string::npos;
            const long long reserve = bare ? (long long) o.vram_reserve_mib : std::strtoll(cmd.c_str() + 4, &end, 10);
            if (!bare && (end == cmd.c_str() + 4 || reserve < 0)) { e = "expected: VRAM [reserve_mib]"; return false; }
            if (!xcache.segmented()) {
                e = "VRAM needs an engine started with --vram-elastic (--serve, one GPU with virtual memory)";
                return false;
            }
            if (!ver.wait_commit(e)) return false;
            apply_pending(true);                     // the adaptive tier's swaps in flight land first
            if (!strata::gpu::device_sync()) { e = std::string("VRAM: ") + strata::gpu::last_error(); return false; }
            const auto t0 = Clock::now();
            size_t free_b = 0, total_b = 0;
            strata::gpu::mem_info(&free_b, &total_b);
            const int64_t want_free = (int64_t) reserve << 20, mapped = xcache.mapped_bytes();
            const int64_t before = xcache.slots();
            // what stays at least: the prompt path's smallest loan (256 tokens) and the 128 slots it must leave
            const int64_t floor_slots =
                std::min<int64_t>(xcache.full_slots(), 128 + (lend_first < 0 ? 0 : lend_slots(256)));
            std::string note;
            if ((int64_t) free_b < want_free) {
                const int64_t floor_b = xcache.bytes_of(floor_slots);
                int64_t keep = mapped - (want_free - (int64_t) free_b);
                if (keep < floor_b) {
                    keep = floor_b;
                    note = " (the cache keeps its smallest size: the prompt path's buffers)";
                }
                if (!xcache.shrink(keep, e)) return false;
                const int64_t live = xcache.slots();
                for (size_t i = 0; i < host_res.size(); ++i)
                    if (host_res[i] >= live) {
                        vram_evicted.emplace_back((int32_t) i, host_res[i]);
                        host_res[i] = strata::core::kNotResident;
                    }
                res_upload();
            } else if (mapped < xcache.full_bytes()) {
                std::string gerr;
                if (!xcache.grow(mapped + ((int64_t) free_b - want_free), gerr)) note = " (" + gerr + ")";
                const int64_t live = xcache.slots();
                std::string ferr;
                std::vector<std::pair<int32_t, int32_t>> keep_out;
                for (const auto& [i, slot] : vram_evicted) {
                    if (slot >= live) { keep_out.emplace_back(i, slot); continue; }
                    const int64_t layer = i / g.n_expert;
                    int64_t pick = i;
                    if (host_res[(size_t) i] >= 0) {   // back already (the adaptive tier): the layer's most-routed miss
                        pick = -1;
                        float best = -1.0f;
                        for (int64_t ex = 0; ex < g.n_expert; ++ex) {
                            const size_t j = (size_t) (layer * g.n_expert + ex);
                            if (host_res[j] >= 0) continue;
                            const float u = drive.d.usage.empty() ? 0.0f : drive.d.usage[j];
                            if (u > best) { best = u; pick = (int64_t) j; }
                        }
                    }
                    if (pick < 0) continue;
                    const uint8_t* b = blob_to_copy(srcp, layer, pick % g.n_expert, fill_tmp);
                    if (b == nullptr || !xcache.fill_slot_queued(slot, b, ferr,
                            (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(layer))) {
                        e = "VRAM: refilling the cache failed: " + ferr;
                        return false;
                    }
                    host_res[(size_t) pick] = slot;
                }
                vram_evicted.swap(keep_out);
                if (!xcache.sync_queued(ferr)) { e = "VRAM: " + ferr; return false; }
                res_upload();
            }
            relend();
            strata::gpu::mem_info(&free_b, &total_b);
            std::fprintf(stderr, "strata serve: VRAM %lld MiB kept free: the expert cache %lld -> %lld of %lld slots "
                                 "(%.2f of %.2f GiB), %lld MiB free now, prompt chunk %lld, in %.0f ms%s\n",
                         reserve, (long long) before, (long long) xcache.slots(), (long long) xcache.full_slots(),
                         (double) xcache.mapped_bytes() / 1073741824.0, (double) xcache.full_bytes() / 1073741824.0,
                         (long long) (free_b >> 20), (long long) o.prefill_chunk,
                         std::chrono::duration<double, std::milli>(Clock::now() - t0).count(), note.c_str());
            std::printf("VRAM reserve_mib=%lld expert_slots=%lld expert_slots_full=%lld expert_cache_mib=%lld "
                        "expert_cache_full_mib=%lld vram_free_mib=%lld prompt_chunk=%lld\n", reserve,
                        (long long) xcache.slots(), (long long) xcache.full_slots(),
                        (long long) (xcache.mapped_bytes() >> 20), (long long) (xcache.full_bytes() >> 20),
                        (long long) (free_b >> 20), (long long) o.prefill_chunk);
            return true;
        };
        // ---- --batch (upstream PR #559, #465): the slots of the batch windows
        struct BSlot {
            bool active = false, stop = false;
            int32_t x = 0;                 ///< the token the next window feeds (not yet in the slot's state)
            int64_t p = 0;                 ///< its position
            int64_t produced = 0, max_new = 0;
            Clock::time_point t0;
            // the slot's own conversation cache (#465): the tokens its session holds (the prompt, then every token a
            // window fed).  Kept when the request ends (`cached`), so the next turn of that conversation continues from
            // it instead of reading its history again (see `slot_source` in the request path).
            std::vector<int32_t> ids;
            bool cached = false;           ///< idle, and its session still holds `ids`
            bool cvec = true;              ///< the control vector setting `ids` were read with
            bool img = false;              ///< the conversation has pictures (never reused from the slot)
            // the checkpoint its prompt read took at the last turn boundary (the history before the new turn's
            // header): a client's next turn renders the history again WITHOUT this reply's thinking, so it matches
            // the slot's tokens only up to there - the slot's K/V up to it plus this state continue from it
            std::vector<ConvCheckpoint> checks;
            // a prompt read that gave way to a waiting request (BYIELD): `ids` is the part read so far, and the same
            // request continues from it (the read goes on with the same chunks; `from0`: it had started at token 0)
            bool partial = false, partial_from0 = false;
        };
        std::vector<BSlot> bs((size_t) std::max(o.batch, 0));
        // timing of the batch windows since the slots were last all idle (one stderr line then)
        double bt_run = 0, bt_commit = 0, bt_emit = 0, bt_wait0 = 0, bt_pool0 = 0;
        int64_t bt_windows = 0, bt_rows = 0;
        Clock::time_point bt_start = Clock::now();
        int admit_slot = -1;               ///< the slot the request being read will continue in (BGEN)
        long long admit_max_new = 0;
        auto batch_on = [&] { for (const BSlot& b : bs) if (b.active) return true; return false; };
        auto try_next_line = [&](std::string& out) -> bool {
            std::lock_guard<std::mutex> lk(in_mu);
            if (in_lines.empty()) return false;
            out = std::move(in_lines.front());
            in_lines.pop_front();
            return true;
        };
        // the session a request just left behind (its prompt and first token) -> slot b's session
        // (a layer split: each GPU's session -> its own slot session, on that GPU)
        auto copy_to_slot = [&](int b, const std::vector<int32_t>& ids, std::string& e) -> bool {
            const int64_t upto = (int64_t) ids.size();
            auto one = [&](int dev, strata::core::SessionState& from, strata::core::SessionState& to) -> bool {
                const strata::core::OnDevice on(dev);
                if (!strata::gpu::device_sync()) { e = "batch admission: device sync failed"; return false; }
                strata::core::ConversationCheckpoint ck;
                ck.ids = ids;
                if (!strata::core::conversation_checkpoint_save(ck, from, g, e) ||
                    !strata::core::conversation_checkpoint_restore(ck, to, g, e))
                    return false;
                for (int64_t j = from.qsa_ord0; j < from.qsa_ord0 + from.qsa_alloc; ++j) {   // the owned layers
                    strata::core::ConversationKv img;
                    if (!strata::core::conversation_kv_save(img, from.qsa_states[j], g, upto, true, e)) return false;
                    if (!strata::gpu::device_sync()) { e = "batch admission: device sync failed"; return false; }
                    if (!strata::core::conversation_kv_restore(img, to.qsa_states[j], g, upto, true, e)) return false;
                }
                if (!strata::gpu::device_sync()) { e = "batch admission: device sync failed"; return false; }
                return true;
            };
            if (!one(0, ss, *bslot_ss[(size_t) b])) return false;
            for (auto& stp : stages)
                if (!one(stp->dev, stp->ss, *stp->bslots[(size_t) b])) return false;
            return true;
        };
        // slot b's session -> the main one (the reverse of copy_to_slot): a request that continues the conversation an
        // idle slot holds reads on from there.  `at`: one of the slot's checkpoints - only the K/V up to it is copied
        // and its state restored instead
        auto copy_from_slot = [&](int b, const ConvCheckpoint* at, std::string& e) -> bool {
            const int64_t upto = at != nullptr ? (int64_t) at->ids.size() : (int64_t) bs[(size_t) b].ids.size();
            strata::core::SessionState& from = *bslot_ss[(size_t) b];
            if (!strata::gpu::device_sync()) { e = "batch slot restore: device sync failed"; return false; }
            if (at != nullptr) {
                if (!strata::core::conversation_checkpoint_restore(*at, ss, g, e)) return false;
            } else {
                strata::core::ConversationCheckpoint ck;
                ck.ids = bs[(size_t) b].ids;
                if (!strata::core::conversation_checkpoint_save(ck, from, g, e) ||
                    !strata::core::conversation_checkpoint_restore(ck, ss, g, e))
                    return false;
            }
            for (int64_t j = from.qsa_ord0; j < from.qsa_ord0 + from.qsa_alloc; ++j) {
                strata::core::ConversationKv img;
                if (!strata::core::conversation_kv_save(img, from.qsa_states[j], g, upto, true, e)) return false;
                if (!strata::gpu::device_sync()) { e = "batch slot restore: device sync failed"; return false; }
                if (!strata::core::conversation_kv_restore(img, ss.qsa_states[j], g, upto, true, e)) return false;
            }
            if (!strata::gpu::device_sync()) { e = "batch slot restore: device sync failed"; return false; }
            return true;
        };
        // one batch window over the active slots only (row t is the t-th active slot): an idle slot is not touched,
        // so it keeps the conversation it holds (#465)
        auto batch_step = [&]() -> bool {
            int S = 0;
            int rows[strata::kernels::kVerifyMaxT] = {};
            int32_t tok[strata::kernels::kVerifyMaxT] = {}, outb[strata::kernels::kVerifyMaxT] = {};
            int64_t pos[strata::kernels::kVerifyMaxT] = {};
            for (int b = 0; b < (int) bs.size() && S < strata::kernels::kVerifyMaxT; ++b)
                if (bs[(size_t) b].active) {
                    rows[S] = b;
                    tok[S] = bs[(size_t) b].x;
                    pos[S] = bs[(size_t) b].p;
                    ++S;
                }
            if (S == 0) return true;
            const bool was_busy = strata::core::progress().busy.load();
            strata::core::progress().busy.store(true);
            drive.d.failed = false;
            apply_pending(false);
            if (bt_windows == 0) {
                bt_start = Clock::now();
                bt_wait0 = ver.ms_wait; bt_pool0 = ver.ms_pool;
            }
            const Clock::time_point w0 = Clock::now();
            if (!ver.run_slot_rows(rows, S, tok, pos, win_pool_fn, win_pool_user, outb, err) || drive.d.failed) {
                std::printf("ERR %s\n", drive.d.failed && drive.d.fail ? drive.d.fail : err.c_str());
                return false;
            }
            const Clock::time_point w1 = Clock::now();
            if (!ver.commit_slots(err)) {
                std::printf("ERR %s\n", err.c_str());
                return false;
            }
            const Clock::time_point w2 = Clock::now();
            auto msd = [](Clock::time_point a0, Clock::time_point b0) { return std::chrono::duration<double, std::milli>(b0 - a0).count(); };
            bt_run += msd(w0, w1);
            bt_commit += msd(w1, w2);
            ++bt_windows;
            bt_rows += S;
            for (int t = 0; t < S; ++t) {
                const int b = rows[t];
                BSlot& sl = bs[(size_t) b];
                const int32_t y = outb[t];
                sl.ids.push_back(sl.x);    // the window fed it: the slot's session holds it now
                std::printf("BT %d %d\n", b, (int) y);
                ++sl.produced;
                const bool eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) y) != o.eos_ids.end();
                const char* fin = eos ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new ? "length"
                                : sl.p + 2 > o.max_context ? "length" : nullptr;
                if (fin != nullptr) {
                    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - sl.t0).count();
                    std::printf("BDONE %d %lld %s %.1f\n", b, (long long) sl.produced, fin, ms);
                    sl.active = false;
                    sl.cached = o.prompt_cache > 0 && !sl.img;   // its session holds sl.ids for the next turn
                } else {
                    sl.x = y;
                    sl.p += 1;
                }
            }
            std::fflush(stdout);
            bt_emit += msd(w2, Clock::now());
            strata::core::progress().busy.store(was_busy);
            if (!batch_on() && bt_windows > 0) {
                const double w = (double) bt_windows, wall = msd(bt_start, Clock::now());
                std::fprintf(stderr, "strata batch: %lld windows, avg %.2f rows, %.2f ms/window = run %.2f (GPU-reach wait "
                                     "%.2f + CPU experts %.2f) + commit %.2f + emit %.2f; %.1f rows/s over %.0f ms of wall "
                                     "time (admissions included)\n",
                             (long long) bt_windows, (double) bt_rows / w, (bt_run + bt_commit + bt_emit) / w, bt_run / w,
                             (ver.ms_wait - bt_wait0) / w, (ver.ms_pool - bt_pool0) / w, bt_commit / w, bt_emit / w,
                             1000.0 * (double) bt_rows / std::max(wall, 1e-9), wall);
                bt_run = bt_commit = bt_emit = 0;
                bt_windows = bt_rows = 0;
            }
            return true;
        };
        // ---- --batch-groups: the slot groups pipelined through the stages (upstream PR #559).  A group's window runs
        // on one stage at a time, its commit right behind it; when it is done there it moves to the next stage, and the
        // stage it left takes the next group.  Each stage on its own GPU, so the GPUs work on different groups at once.
        const int n_pipe = n_stages;
        if (o.batch > 0 && o.batch_groups > 1 && (n_pipe < 2 || split_same || o.batch % o.batch_groups != 0)) {
            std::fprintf(stderr, "strata serve: --batch-groups %d is off: it needs a layer split over GPUs of their "
                                 "own and to divide --batch %d\n", o.batch_groups, o.batch);
            o.batch_groups = 1;
        }
        const bool piped = o.batch > 0 && o.batch_groups > 1;
        const int GS = piped ? o.batch / o.batch_groups : o.batch;
        // --batch-cpu-split (experimental): stage 1's CPU experts through a Drive and a pool of their own, and each
        // stage's windows served by a host thread of its own, so the two stages' CPU work runs at once
        const bool pipe_threads = piped && pool1 != nullptr && n_pipe == 2;
        Drive drive1 = drive;
        drive1.d.pool = pool1.get();
        SplitDrive split_drive1 = split_drive;
        split_drive1.base = &drive1;
        struct StagePoll {
            std::thread th;
            std::mutex mu;
            std::condition_variable cv;
            bool go = false, quit = false;
            std::atomic<int> result{2};   // 0 running, 1 done, -1 failed, 2 idle
            std::string err;
        };
        std::vector<std::unique_ptr<StagePoll>> spoll;
        struct StagePollJoin {
            std::vector<std::unique_ptr<StagePoll>>& v;
            ~StagePollJoin() {
                for (auto& sp : v) {
                    { std::lock_guard<std::mutex> lk(sp->mu); sp->quit = true; }
                    sp->cv.notify_all();
                    if (sp->th.joinable()) sp->th.join();
                }
            }
        } spoll_join{spoll};
        if (pipe_threads)
            for (int k = 0; k < n_pipe; ++k) {
                spoll.push_back(std::make_unique<StagePoll>());
                StagePoll* sp = spoll.back().get();
                void* user = k == 0 ? (void*) &split_drive : (void*) &split_drive1;
                sp->th = std::thread([&, sp, k, user] {
                    for (;;) {
                        {
                            std::unique_lock<std::mutex> lk(sp->mu);
                            sp->cv.wait(lk, [sp] { return sp->go || sp->quit; });
                            if (sp->quit) return;
                            sp->go = false;
                        }
                        int rr;
                        while ((rr = stage_ver(k).batch_poll(win_pool_fn, user, sp->err)) == 0)
                            std::this_thread::sleep_for(std::chrono::microseconds(20));
                        sp->result.store(rr);
                    }
                });
            }
        struct PGroup {
            bool inflight = false;
            int stage = 0;                  ///< the stage it runs on or waits for
            int32_t tok[strata::kernels::kVerifyMaxT] = {};
            int64_t pos[strata::kernels::kVerifyMaxT] = {};
            int64_t since = 0;              ///< the turn it started waiting (the longest waiting goes first)
            int S = 0;                      ///< slots in its window: up to its last active one (idle ones cost rows)
        };
        std::vector<PGroup> pg((size_t) (piped ? o.batch_groups : 0));
        std::vector<int> stage_group((size_t) n_pipe, -1);
        int64_t pipe_tick = 0, rr = 0;
        std::vector<double> pipe_pool0((size_t) n_pipe, 0.0);   // each stage's CPU-expert time at the run's start
        auto pipe_inflight = [&] { for (const PGroup& x : pg) if (x.inflight) return true; return false; };
        auto group_active = [&](int gi) {
            for (int t = 0; t < GS; ++t) if (bs[(size_t) gi * (size_t) GS + (size_t) t].active) return true;
            return false;
        };
        if (piped)
            std::fprintf(stderr, "strata serve: --batch %d in %d groups of %d, pipelined through %d GPUs\n", o.batch,
                         o.batch_groups, GS, n_pipe);
        // one turn: serve every running stage, hand finished groups on, start what can start
        auto pump = [&](bool may_start) -> bool {
            ++pipe_tick;
            if (bt_windows == 0 && !pipe_inflight()) {
                bt_start = Clock::now();
                for (int k = 0; k < n_pipe; ++k) pipe_pool0[(size_t) k] = stage_ver(k).ms_pool;
            }
            for (int k = 0; k < n_pipe; ++k) {
                const int gi = stage_group[(size_t) k];
                if (gi < 0) continue;
                int r;
                if (pipe_threads) {
                    r = spoll[(size_t) k]->result.load();
                    if (r < 0) err = spoll[(size_t) k]->err;
                    if (r == 1) spoll[(size_t) k]->result.store(2);
                } else {
                    r = stage_ver(k).batch_poll(win_pool_fn, win_pool_user, err);
                }
                if (r < 0) { std::printf("ERR %s\n", err.c_str()); return false; }
                if (r == 0) continue;
                stage_group[(size_t) k] = -1;
                PGroup& G = pg[(size_t) gi];
                if (k + 1 < n_pipe) { G.stage = k + 1; G.since = pipe_tick; continue; }
                const int32_t* outb = stage_ver(k).batch_out();   // the last stage: the group's picks
                for (int t = 0; t < G.S; ++t) {
                    const int b = gi * GS + t;
                    BSlot& sl = bs[(size_t) b];
                    if (!sl.active) continue;
                    const int32_t y = outb[t];
                    sl.ids.push_back(sl.x);    // the window fed it: the slot's session holds it now
                    std::printf("BT %d %d\n", b, (int) y);
                    ++sl.produced;
                    ++bt_rows;
                    const bool eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) y) != o.eos_ids.end();
                    const char* fin = eos ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new ? "length"
                                    : sl.p + 2 > o.max_context ? "length" : nullptr;
                    if (fin != nullptr) {
                        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - sl.t0).count();
                        std::printf("BDONE %d %lld %s %.1f\n", b, (long long) sl.produced, fin, ms);
                        sl.active = false;
                        sl.cached = false;   // the group's pad rows write an idle slot's state: not reused as a cache
                    } else {
                        sl.x = y;
                        sl.p += 1;
                    }
                }
                std::fflush(stdout);
                ++bt_windows;
                G.inflight = false;
            }
            // start waiting groups on free stages, the longest waiting first; a new step on stage 0 round-robin
            for (int k = n_pipe - 1; k >= 0; --k) {
                if (stage_group[(size_t) k] >= 0) continue;
                int pick = -1;
                if (k > 0) {
                    for (int gi = 0; gi < (int) pg.size(); ++gi)
                        if (pg[(size_t) gi].inflight && pg[(size_t) gi].stage == k &&
                            (pick < 0 || pg[(size_t) gi].since < pg[(size_t) pick].since))
                            pick = gi;
                    if (pick < 0) continue;
                } else {
                    if (!may_start) continue;
                    for (int j = 0; j < (int) pg.size() && pick < 0; ++j) {
                        const int gi = (int) ((rr + j) % (int64_t) pg.size());
                        if (!pg[(size_t) gi].inflight && group_active(gi)) pick = gi;
                    }
                    if (pick < 0) continue;
                    rr = pick + 1;
                    PGroup& G = pg[(size_t) pick];
                    G.S = 0;
                    for (int t = 0; t < GS; ++t) if (bs[(size_t) pick * (size_t) GS + (size_t) t].active) G.S = t + 1;
                    for (int t = 0; t < G.S; ++t) {
                        BSlot& sl = bs[(size_t) pick * (size_t) GS + (size_t) t];
                        G.tok[t] = sl.active ? sl.x : 0;
                        G.pos[t] = sl.active ? sl.p : 0;
                        if (!sl.active) sl.cached = false;   // its pad row writes its state
                    }
                    G.inflight = true;
                    G.stage = 0;
                    drive.d.failed = false;
                    apply_pending(false);
                }
                PGroup& G = pg[(size_t) pick];
                strata::core::progress().busy.store(true);
                if (!stage_ver(k).batch_launch(pick * GS, G.S, G.tok, G.pos, err)) {
                    std::printf("ERR %s\n", err.c_str());
                    return false;
                }
                stage_group[(size_t) k] = pick;
                if (pipe_threads) {   // the stage's thread serves the window from here
                    StagePoll& sp = *spoll[(size_t) k];
                    sp.result.store(0);
                    { std::lock_guard<std::mutex> lk(sp.mu); sp.go = true; }
                    sp.cv.notify_one();
                }
            }
            if (drive.d.failed || drive1.d.failed) {
                const char* f = drive.d.failed ? drive.d.fail : drive1.d.fail;
                std::printf("ERR %s\n", f ? f : "the expert pool failed");
                return false;
            }
            if (!pipe_inflight() && !batch_on() && bt_windows > 0) {   // all idle: one timing line
                const double wall = std::chrono::duration<double, std::milli>(Clock::now() - bt_start).count();
                std::fprintf(stderr,
                             "strata batch (pipelined, %d groups of %d): %lld group-steps, %lld rows in %.0f ms = %.1f "
                             "rows/s (admissions included)\n", o.batch_groups, GS, (long long) bt_windows,
                             (long long) bt_rows, wall, 1000.0 * (double) bt_rows / std::max(wall, 1e-9));
                // the host thread serves every stage's CPU experts in turn: how much of the wall time that took
                std::string pools;
                for (int k = 0; k < n_pipe; ++k)
                    pools += " " + std::to_string((int) (stage_ver(k).ms_pool - pipe_pool0[(size_t) k])) + " ms";
                std::fprintf(stderr, "strata batch (pipelined): CPU experts per stage:%s of %.0f ms\n", pools.c_str(),
                             wall);
                if (pipe_threads) {   // stage 1's routing counts go back to the one the adaptive tier reads
                    for (size_t i = 0; i < drive.d.usage.size() && i < drive1.d.usage.size(); ++i) {
                        drive.d.usage[i] += drive1.d.usage[i];
                        drive1.d.usage[i] = 0.0f;
                    }
                }
                bt_windows = bt_rows = 0;
                strata::core::progress().busy.store(false);
            }
            return true;
        };
        auto pipe_drain = [&]() -> bool {
            while (pipe_inflight())
                if (!pump(false)) return false;
            return true;
        };
        for (;;) {
            if (batch_on() || (piped && pipe_inflight())) {   // the slots decode while no line waits
                if (!try_next_line(line)) {
                    if (!(piped ? pump(true) : batch_step())) return 1;
                    // the stages' threads poll; spinning here slowed them 3x (it starved the runtime's threads)
                    if (pipe_threads) std::this_thread::sleep_for(std::chrono::microseconds(20));
                    continue;
                }
                // a request reads its prompt through every stage: the groups in flight finish first
                if (piped && line.rfind("BSTOP ", 0) != 0 && !pipe_drain()) return 1;
            } else if (!next_line(line)) {
                break;
            }
            // --batch: BSTOP <slot> ends that slot at its next batch window; BGEN <slot> <max_new> ... reads the
            // request's prompt and first token as a GEN 1, then continues it in that slot
            if (line.rfind("BSTOP ", 0) == 0) {
                const int b = (int) std::strtol(line.c_str() + 6, nullptr, 10);
                if (b >= 0 && b < (int) bs.size()) bs[(size_t) b].stop = true;
                continue;
            }
            if (line.rfind("BYIELD", 0) == 0) continue;   // for a prompt read that has ended meanwhile
            admit_slot = -1;
            if (line.rfind("BGEN ", 0) == 0 || line.rfind("BGENI ", 0) == 0) {
                const bool bimg = line.rfind("BGENI ", 0) == 0;
                char* e1 = nullptr;
                const long b = std::strtol(line.c_str() + (bimg ? 6 : 5), &e1, 10);
                char* e2 = nullptr;
                const long long mn = std::strtoll(e1, &e2, 10);
                if (bs.empty() || b < 0 || b >= (long) bs.size() || bs[(size_t) b].active || mn < 1 || e2 == e1) {
                    std::printf("ERR BGEN: no such free slot (--batch %d) or a bad max_new\n", o.batch);
                    std::fflush(stdout);
                    continue;
                }
                admit_slot = (int) b;
                admit_max_new = mn;
                line = std::string(bimg ? "GENI 1" : "GEN 1") + e2;
            }
            // #477: every --expert-profile-save-every minutes, before the next request (at QUIT: after the loop)
            if (!heat.empty() && line != "QUIT" && o.expert_profile_save_min > 0 &&
                Clock::now() - profile_saved_at >= std::chrono::duration<double>(o.expert_profile_save_min * 60.0))
                save_profile("periodic");
            // --conversation-save: files past their age go between requests too, not only at the next start
            if (disk.enabled() && line != "QUIT" && Clock::now() - disk_expired_at >= std::chrono::minutes(10)) {
                if (const size_t n = disk.expire())
                    std::fprintf(stderr, "strata serve: conversation save: deleted %zu old file%s\n", n, n == 1 ? "" : "s");
                disk_expired_at = Clock::now();
            }
            if (line == "QUIT") break;
            if (line.rfind("VRAM", 0) == 0) {   // #533 (above): between requests, not a request
                std::string verr;
                if (batch_on()) verr = "VRAM: not while batch slots are decoding";
                else vram_command(line, verr);
                if (!verr.empty()) std::printf("ERR %s\n", verr.c_str());
                std::fflush(stdout);
                continue;
            }
            // the watchdog watches a request from here until this iteration ends, whichever way it ends
            struct BusyScope {
                BusyScope() {
                    strata::core::progress().allow_until_ms.store(0);
                    strata::core::progress().busy.store(true);
                    strata::core::progress_at("request");
                }
                ~BusyScope() {
                    strata::core::progress().busy.store(false);
                    strata::core::progress().allow_until_ms.store(0);
                    strata::core::progress_at("idle");
                }
            } busy_scope;
            stop_req.store(false);   // a STOP that arrived between requests is stale
            err.clear();               // nor may a cancelled request's error fail this one (upstream 6cad1fe)
            // Disk sessions: SAVE <path> | RESTORE <path>, between requests (the path runs to the end of the line,
            // UTF-8).  The file holds what a parked conversation holds (conversation_file.hpp).  Answers: SAVED /
            // RESTORED <tokens> <bytes> <ms>; SERR <invalid|storage|memory|io> <published 0|1> <reason> for a refusal
            // or failure that leaves the engine in step (the live session as it was, unless published=1 says the new
            // file already replaced the old one); FATAL <reason> (and the engine exits) for a restore transfer that
            // failed after the device state was touched.  On the way: SESSION <done> <total> after every block of
            // the file moved (at most 16 MiB), and SWAIT <phase> <seconds> before a step that blocks in one call
            // (fingerprint file, state capture, file flush, rename + folder flush, validation, device transfer):
            // that step is allowed <seconds> (session_phase_limit_s) by the watchdog and by the server, no more.
            // (A verifier commit that fails before the command still answers ERR and exits, as for a request.)
            if (line.rfind("SAVE ", 0) == 0 || line.rfind("RESTORE ", 0) == 0) {
                const bool save = line[0] == 'S';
                const std::string path = line.substr(save ? 5 : 8);
                const auto t0 = Clock::now();
                auto ms = [&] { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); };
                auto refuse = [&](const std::string& why, strata::core::SessionError kind = strata::core::SessionError::invalid,
                                  bool published = false) {
                    std::fprintf(stderr, "strata serve: session %s %s: %s\n", save ? "save" : "restore", path.c_str(),
                                 why.c_str());
                    std::string one = why;   // one protocol line
                    for (char& c : one) if (c == '\n' || c == '\r') c = ' ';
                    std::printf("SERR %s %d %s\n", strata::core::session_error_name(kind), published ? 1 : 0, one.c_str());
                    std::fflush(stdout);
                    err.clear();
                };
                if (!use_mtp) {   // a session file carries the draft layer's K/V, as a parked conversation does
                    refuse("session files need --mtp");
                    continue;
                }
                auto moving = [&](uint64_t done, uint64_t total) {
                    strata::core::progress_at(save ? "writing a session file, MiB" : "reading a session file, MiB",
                                              (int64_t) (done >> 20));
                    strata::core::progress_allow(0);   // a block moved: the ordinary watchdog limit again, and a beat
                    std::printf("SESSION %llu %llu\n", (unsigned long long) done, (unsigned long long) total);
                    std::fflush(stdout);
                };
                // a step that blocks in one call: its explicit allowance, on both sides, announced before it starts
                auto blocking = [&](const char* phase, uint64_t bytes) {
                    const int64_t s = strata::core::session_phase_limit_s(bytes);
                    strata::core::progress_at(phase);
                    strata::core::progress_allow(s);
                    std::printf("SWAIT %s %lld\n", phase, (long long) s);
                    std::fflush(stdout);
                };
                strata::core::progress_at(save ? "saving a session" : "restoring a session");
                if (path.empty()) { refuse("missing path"); continue; }
                if (!stages.empty() || multi_gpu) { refuse("session files do not support --layer-split"); continue; }
                if (o.prompt_cache <= 0) { refuse("session files need --prompt-cache > 0"); continue; }
                if (!ver.wait_commit(err)) {
                    std::printf("ERR %s\n", err.c_str());
                    return 1;
                }
                strata::core::SessionFileIdentity id;
                try {
                    if (!session_identity(id, err, [&] { blocking("fingerprint", 2u << 20); })) {
                        refuse(err, strata::core::SessionError::io);
                        continue;
                    }
                } catch (const std::bad_alloc&) {
                    refuse("session identity: out of memory", strata::core::SessionError::memory);
                    continue;
                } catch (const std::exception& e) {
                    refuse(std::string("session identity: ") + e.what(), strata::core::SessionError::io);
                    continue;
                }
                if (save) {
                    if (!live_ok || live.empty()) { refuse("no complete session to save"); continue; }
                    size_t bytes = 0, kept = 0;
                    double capture_ms = 0;
                    try {
                        // RAM PREFLIGHT for the save (not a reservation), BEFORE anything is copied: the deepest
                        // checkpoint is chosen by reference, and the copies it will take (the list below and the one
                        // in `meta`), the copied live running state with its tokens and images, the K/V source
                        // directory and the 16 MiB write buffer are asked for above the parking floor.  Only the
                        // deepest checkpoint goes to disk (the next turn's resume point); the K/V is streamed from
                        // the authoritative pools straight into the file, without a full host capture.
                        strata::core::SessionSaveLive sl;
                        {
                            strata::core::ConversationStateSizes z;
                            std::string why;
                            uint64_t state = UINT64_MAX;   // unknown sizes: the preflight refuses rather than guesses
                            if (strata::core::conversation_session_sizes(g, ss, z, why)) {
                                const uint64_t q = (uint64_t) std::max<int64_t>(ss.qsa_alloc, 0);
                                const uint64_t per = (uint64_t) z.tail + z.dead + z.block_pos;
                                if (q == 0 || per <= (UINT64_MAX - z.gdn - z.ple) / q) state = z.gdn + z.ple + q * per;
                            }
                            sl.state_bytes = state;
                            sl.tokens = live.size();
                            sl.images = live_imgs.size();
                            sl.kv_layers = (uint64_t) std::max<int64_t>(ss.qsa_alloc, 0) + 1;
                        }
                        const uint64_t floor = (uint64_t) o.conversation_cache_min_free_mib << 20;
                        auto admit = [&](uint64_t need, std::string& why) {
                            const auto avail = strata::core::conversation_available_memory();
                            if (strata::core::conversation_memory_admit(avail, need, floor)) return true;
                            why = "not enough RAM to save the session (" +
                                  (need == UINT64_MAX ? std::string("unknown") : std::to_string(need >> 20)) +
                                  " MiB plus a floor of " + std::to_string((long long) o.conversation_cache_min_free_mib) +
                                  " MiB needed, " + (avail ? std::to_string(*avail >> 20) + " MiB available)"
                                                           : "RAM telemetry unavailable)");
                            return false;
                        };
                        std::vector<ConvCheckpoint> disk_checks;
                        std::string why;
                        if (!strata::core::session_save_checkpoints(checks, sl, admit, disk_checks, why)) {
                            refuse(why, strata::core::SessionError::memory);
                            continue;
                        }
                        kept = disk_checks.size();
                        const strata::core::ConversationView view{live, live_imgs, disk_checks, cvec_cached};
                        strata::core::SavedConversation meta;
                        std::vector<strata::core::SessionKvSource> sources;
                        // the live running state comes off the device in one synchronous copy
                        blocking("capture", sl.state_bytes == UINT64_MAX ? 0 : sl.state_bytes);
                        if (!strata::core::conversation_snapshot_sources(meta, sources, view, ss, g, mtp.kv_state(),
                                                                         err)) {
                            refuse(err, strata::core::SessionError::io);
                            continue;
                        }
                        capture_ms = ms();
                        strata::core::SessionWriteOptions wo;
                        wo.min_free_bytes = (uint64_t) o.session_min_free_mib << 20;
                        wo.progress = moving;
                        wo.phase = blocking;
                        strata::core::SessionStatus st;
                        if (!strata::core::session_file_write(path, meta, sources, id, bytes, err, wo, &st)) {
                            refuse(err, st.error, st.published);
                            continue;
                        }
                        if (st.dir_flush_unsupported)
                            std::fprintf(stderr, "strata serve: session saved to %s; this filesystem cannot flush a "
                                                 "folder, so the new name is not flushed\n", path.c_str());
                    } catch (const std::bad_alloc&) {
                        refuse("not enough RAM to save the session", strata::core::SessionError::memory);
                        continue;
                    }
                    std::fprintf(stderr, "strata serve: session saved %zu tokens, %zu of %zu checkpoints, %zu bytes to %s "
                                 "in %.1f ms (state %.1f ms)\n", live.size(), kept, checks.size(), bytes,
                                 path.c_str(), ms(), capture_ms);
                    std::printf("SAVED %zu %zu %.1f\n", live.size(), bytes, ms());
                } else {
                    strata::core::SavedConversation image;
                    size_t bytes = 0;
                    try {
                        // bounds this session can ever restore (geometry, layer range, tokens, checkpoints, every
                        // state and K/V part), checked as the file is parsed and before each array is allocated; the
                        // file size follows from them.  The RAM the parse needs at its peak is asked first.
                        strata::core::SessionReadLimits limits;
                        limits.progress = moving;
                        const uint64_t floor = (uint64_t) o.conversation_cache_min_free_mib << 20;
                        limits.admit = [floor, &o](uint64_t need, std::string& why) {
                            const auto avail = strata::core::conversation_available_memory();
                            if (strata::core::conversation_memory_admit(avail, need, floor)) return true;
                            why = "not enough RAM to read it (" + std::to_string(need >> 20) + " MiB plus a floor of " +
                                  std::to_string((long long) o.conversation_cache_min_free_mib) + " MiB needed, " +
                                  (avail ? std::to_string(*avail >> 20) + " MiB available)" : "RAM telemetry unavailable)");
                            return false;
                        };
                        if (!strata::core::conversation_session_read_limits(
                                limits, ss, g, mtp.kv_state(), (uint64_t) o.max_context,
                                (uint64_t) std::max(o.prompt_cache, 1), err)) {
                            refuse(err, strata::core::SessionError::io);
                            continue;
                        }
                        strata::core::SessionStatus st;
                        if (!strata::core::session_file_read(path, id, image, bytes, err, limits, &st)) {
                            refuse(err, st.error);
                            continue;
                        }
                    } catch (const std::bad_alloc&) {
                        refuse("not enough RAM to read the session", strata::core::SessionError::memory);
                        continue;
                    }
                    const double read_ms = ms();
                    // the whole image against this engine, still without any device write
                    blocking("validate", bytes);
                    if (!strata::core::conversation_snapshot_validate(image, ss, g, mtp.kv_state(), err)) {
                        refuse(err);
                        continue;
                    }
                    conversations.take_reuse();   // retained K/V described the outgoing session
                    live_ok = false;
                    // host -> device in synchronous copies of the whole state: one bounded allowance
                    blocking("transfer", bytes);
                    if (strata::core::conversation_snapshot_restore(image, ss, g, mtp.kv_state(), err) !=
                        strata::core::ConversationRestore::restored) {
                        // validated above: a failure here is a transfer failure, after device writes began - never
                        // decode from a partial state; the server starts the engine again
                        std::fprintf(stderr, "strata serve: session restore %s: transfer failed: %s\n", path.c_str(),
                                     err.c_str());
                        std::printf("FATAL restoring the session file failed after the device state was changed: %s\n",
                                    err.c_str());
                        std::fflush(stdout);
                        return 1;
                    }
                    live = std::move(image.live.ids);
                    live_imgs = std::move(image.live.imgs);
                    checks = std::move(image.checkpoints);
                    for (const ConvCheckpoint& c : checks) check_clock = std::max(check_clock, c.used);
                    cvec_cached = image.cvec;
                    live_ok = true;
                    std::fprintf(stderr, "strata serve: session restored %zu tokens, %zu checkpoints, %zu bytes from %s "
                                 "in %.1f ms (read+check %.1f ms)\n", live.size(), checks.size(), bytes, path.c_str(),
                                 ms(), read_ms);
                    std::printf("RESTORED %zu %zu %.1f\n", live.size(), bytes, ms());
                }
                std::fflush(stdout);
                continue;
            }
            const bool geni = line.rfind("GENI ", 0) == 0;
            if (!geni && line.rfind("GEN ", 0) != 0) {
                std::printf("ERR expected: GEN <max_new> <id,id,...> or GENI <max_new> <file> <id,id,...>\n");
                continue;
            }
            char* endp = nullptr;
            const long long max_new = std::strtoll(line.c_str() + (geni ? 5 : 4), &endp, 10);
            // optional sampling keys between max_new and the ids: temperature=F, top_p=F, top_k=N, min_p=F,
            // penalty_last_n=N, penalty_repeat=F, penalty_freq=F, penalty_present=F, seed=N (text requests
            // only).  Absent keys keep today's behavior: greedy, no penalties.
            float req_temperature = 0.0f, req_top_p = 1.0f;
            int req_top_k = 20;   // the sampler's own default; the sampled path REQUIRES top_k in 1..64
            unsigned long long req_seed = 0;
            float req_min_p = 0.0f, req_penalty_repeat = 1.0f, req_penalty_freq = 0.0f, req_penalty_present = 0.0f;
            int req_penalty_last_n = 0;
            int req_cvec = 1;   // cvec=0|1: a loaded control vector for this request (on when absent)
            // ckpt=0: a one-shot call whose turn no later request extends.  No checkpoint at its last turn boundary
            // (so no split there) nor every --prompt-cache-every tokens, and its session is neither continued nor
            // parked after it.  It still resumes from a checkpoint it matches, and still saves the system-prompt root
            // when that reaches --prompt-cache-root.  Absent = checkpointed as before.
            int req_ckpt = 1;
            // tuning keys (setup's calibration measures settings without restarting the engine): the PCIe share of
            // the missed experts and the draft-probability floor, for this request only
            double req_pcie_frac = o.pcie_frac, req_spec_min_p = o.spec_min_p;
            // logprobs=K (0..20; upstream's Intel port 4ba35fd): after each "T id" an "LP logprob id:logprob ..." line
            // with the token's log-probability and the K most likely tokens', from the verify window's head logits
            // (before sampling, penalties and temperature).  -1 (absent): no LP lines
            int req_logprobs = -1;
            if (endp != nullptr) {   // GENI takes the same keys (#75: image requests were always greedy); its
                                     // embedding file path is the first token without an =
                for (;;) {
                    while (*endp == ' ') ++endp;
                    const char* start = endp;
                    while (*endp != '\0' && *endp != ' ') ++endp;
                    if (endp == start) break;
                    const std::string tok(start, (size_t) (endp - start));
                    const size_t eq = tok.find('=');
                    if (eq == std::string::npos) { endp = const_cast<char*>(start); break; }
                    const std::string key = tok.substr(0, eq);
                    const float fv = std::strtof(tok.c_str() + eq + 1, nullptr);
                    if (key == "cvec") req_cvec = std::atoi(tok.c_str() + eq + 1);
                    else if (key == "ckpt") req_ckpt = std::strtol(tok.c_str() + eq + 1, nullptr, 10) != 0;
                    else if (key == "temperature") req_temperature = fv;
                    else if (key == "top_p") req_top_p = fv;
                    else if (key == "top_k") req_top_k = std::atoi(tok.c_str() + eq + 1);
                    else if (key == "min_p") req_min_p = fv;
                    else if (key == "penalty_last_n") req_penalty_last_n = std::atoi(tok.c_str() + eq + 1);
                    else if (key == "logprobs") req_logprobs = (int) std::clamp(std::strtol(tok.c_str() + eq + 1, nullptr, 10), -1L, 20L);
                    else if (key == "penalty_repeat") req_penalty_repeat = fv;
                    else if (key == "penalty_freq") req_penalty_freq = fv;
                    else if (key == "penalty_present") req_penalty_present = fv;
                    else if (key == "seed") req_seed = std::strtoull(tok.c_str() + eq + 1, nullptr, 10);
                    else if (key == "pcie_frac") req_pcie_frac = std::clamp((double) fv, 0.0, 1.0);
                    else if (key == "spec_min_p") req_spec_min_p = std::clamp((double) fv, 0.0, 1.0);
                    // unknown keys are skipped: the ids start at the first token without '='
                }
            }
            std::string emb_path;
            if (geni && endp != nullptr) {
                while (*endp == ' ') ++endp;
                char* gap = std::strchr(endp, ' ');
                if (gap != nullptr) { emb_path.assign(endp, (size_t) (gap - endp)); endp = gap; }
            }
            std::vector<int64_t> ids;
            std::string pe;
            if (max_new < 1 || endp == nullptr || (geni && emb_path.empty()) || !parse_i64_list(endp, ids, pe)) {
                std::printf("ERR bad request: %s\n", pe.empty() ? "max_new" : pe.c_str());
                continue;
            }
            const int64_t n = (int64_t) ids.size();
            req_imgs.clear();
            if (geni && !o.vision) { std::printf("ERR this engine was started without --vision\n"); continue; }
            if (geni || !mrope_identity) {
                // positions for every cell this request can reach; the identity again for a text request
                std::string ve;
                row_ptr.assign((size_t) n, nullptr);
                const int64_t cells = (int64_t) mrope_host.size() / 3;
                auto put = [&](int64_t c, int64_t t, int64_t h, int64_t w) {
                    mrope_host[(size_t) c * 3] = (int32_t) t;
                    mrope_host[(size_t) c * 3 + 1] = (int32_t) h;
                    mrope_host[(size_t) c * 3 + 2] = (int32_t) w;
                };
                if (!geni) {
                    for (int64_t c = 0; c < cells; ++c) put(c, c, c, c);
                } else {
                    struct Img { int64_t n, nx, ny; size_t off; };
                    std::vector<Img> imgs;
                    img_rows.clear();
                    std::FILE* f = std::fopen(emb_path.c_str(), "rb");
                    if (!f) ve = "cannot open " + emb_path;
                    while (f && ve.empty()) {
                        int32_t hdr[5];
                        const size_t got = std::fread(hdr, sizeof(int32_t), 5, f);
                        if (got == 0) break;
                        if (got != 5 || hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 ||
                            (int64_t) hdr[2] * hdr[3] != hdr[1] || hdr[4] != (int32_t) g.n_embd) {
                            ve = "bad embeddings file (expected strata-vision records of width " +
                                 std::to_string((long long) g.n_embd) + ")";
                            break;
                        }
                        const size_t off = img_rows.size(), cnt = (size_t) hdr[1] * (size_t) hdr[4];
                        img_rows.resize(off + cnt);
                        if (std::fread(img_rows.data() + off, sizeof(float), cnt, f) != cnt) { ve = "short embeddings file"; break; }
                        imgs.push_back({hdr[1], hdr[2], hdr[3], off});
                    }
                    if (f) std::fclose(f);
                    int64_t p = 0, i = 0;
                    size_t k = 0;
                    while (ve.empty() && i < n) {
                        if (ids[(size_t) i] != kImagePad) { put(i, p, p, p); ++p; ++i; continue; }
                        if (k >= imgs.size()) { ve = "the prompt has more images than the embeddings file"; break; }
                        const Img& im = imgs[k++];
                        {   // what the conversation cache compares: a picture is its grid and its embeddings
                            const int64_t grid[3] = {im.n, im.nx, im.ny};
                            uint64_t h = fnv1a(grid, sizeof grid);
                            h = fnv1a(img_rows.data() + im.off, (size_t) im.n * (size_t) g.n_embd * sizeof(float), h);
                            req_imgs.push_back({i, h});
                        }
                        for (int64_t j = 0; j < im.n && ve.empty(); ++j)
                            if (i + j >= n || ids[(size_t) (i + j)] != kImagePad)
                                ve = "image " + std::to_string(k) + " has " + std::to_string((long long) im.n) +
                                     " rows but fewer <|image_pad|> tokens";
                        for (int64_t j = 0; j < im.n && ve.empty(); ++j) {
                            const int64_t y = j / im.nx, x = j % im.nx;
                            put(i + j, p, p + y, p + x);
                            row_ptr[(size_t) (i + j)] = img_rows.data() + im.off + (size_t) j * (size_t) g.n_embd;
                        }
                        i += im.n;
                        p += std::max(im.nx, im.ny);
                    }
                    if (ve.empty() && k != imgs.size()) ve = "the embeddings file has more images than the prompt";
                    if (ve.empty() && n > 0 && ids[(size_t) (n - 1)] == kImagePad) ve = "the prompt cannot end in an image";
                    for (int64_t c = n; ve.empty() && c < cells; ++c) put(c, p + (c - n), p + (c - n), p + (c - n));
                }
                tr("positions built", (long long) img_rows.size());
                strata::gpu::device_sync();
                tr("device idle");
                // CUDA0's table and, with a layer split, every later stage's (each device reads its own)
                auto upload_mrope = [&]() -> bool {
                    bool ok = strata::gpu::copy(d_mrope, mrope_host.data(), mrope_host.size() * sizeof(int32_t));
                    for (auto& st : stages) {
                        const strata::core::OnDevice on(st->dev);
                        strata::gpu::device_sync();
                        ok = ok && strata::gpu::copy(st->mrope, mrope_host.data(), mrope_host.size() * sizeof(int32_t));
                    }
                    return ok;
                };
                if (ve.empty() && !upload_mrope()) ve = "the image position upload failed";
                if (!ve.empty()) {
                    // leave the table as the identity so the next text request is untouched
                    for (int64_t c = 0; c < cells; ++c) put(c, c, c, c);
                    upload_mrope();
                    mrope_identity = true;
                    std::printf("ERR %s\n", ve.c_str());
                    std::fflush(stdout);
                    continue;
                }
                mrope_identity = !geni;
            }
            sp.embd_rows = geni ? row_ptr.data() : nullptr;
            if (n + max_new + 8 > o.max_context) {
                std::printf("ERR prompt (%lld tokens) + max_new (%lld) exceeds the context (%lld)\n", (long long) n,
                            (long long) max_new, (long long) o.max_context);
                continue;
            }
            bool bad = false;
            for (int64_t t : ids) bad = bad || t < 0 || t >= n_vocab;
            if (bad) { std::printf("ERR a token id is outside the vocabulary\n"); continue; }
            std::array<int64_t, 3> remote_before{};
            std::array<int64_t, 3> launches_before{};
            std::array<uint64_t, 3> compact_before{}, full_before{};
            for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0)
            {
                remote_before[(size_t) r] = remote_experts[(size_t) r].computed();
                launches_before[(size_t) r] = remote_experts[(size_t) r].launched_layers();
                compact_before[(size_t) r] = remote_experts[(size_t) r].returned_bytes();
                full_before[(size_t) r] = remote_experts[(size_t) r].full_row_bytes();
            }
            cur = ids;
            const Clock::time_point r0 = Clock::now();
            // ---- where this request starts reading: the live session, or a checkpoint, whose tokens AND pictures are
            // exactly the start of this prompt - at most n - 1 of them, the last token is always the first window
            auto starts_with = [&](const std::vector<int32_t>& pre, const std::vector<ImgKey>& pre_imgs) -> bool {
                const int64_t L = (int64_t) pre.size();
                if (L < 1 || L > n - 1) return false;
                for (int64_t i = 0; i < L; ++i)
                    if ((int32_t) ids[(size_t) i] != pre[(size_t) i]) return false;
                return imgs_below(req_imgs, L) == pre_imgs;
            };
            const bool want_cvec = strata::kernels::cvec().loaded() ? req_cvec != 0 : true;
            // the last request's final commit may still run on the verifier's queue (set_commit_async): everything
            // below reads, restores or zeroes the session from other queues and the host
            if (!ver.wait_commit(err)) {
                std::printf("ERR %s\n", err.c_str());
                return 1;
            }
            int64_t resume = 0;
            bool from_live = false;
            // the live session and the checkpoints were read with the control vector one way: a switch reads the
            // prompt again from the start (or from a parked conversation read the other way)
            if (o.prompt_cache > 0 && want_cvec == cvec_cached) {
                if (live_ok && starts_with(live, live_imgs)) { resume = (int64_t) live.size(); from_live = true; }
                for (const ConvCheckpoint& c : checks)
                    if ((int64_t) c.ids.size() > resume && starts_with(c.ids, c.imgs)) {
                        resume = (int64_t) c.ids.size();
                        from_live = false;
                    }
            }
            // --batch: an idle slot that holds the start of this prompt (the conversation it served last) is a source
            // too - its session is copied back below, so only the new part is read (#465)
            int slot_source = -1;
            int64_t slot_tokens = 0;
            bool resumed_from0 = false;   // a prompt read parked in a slot (BYIELD) that had started at token 0
            const ConvCheckpoint* slot_ck = nullptr;   // the slot's checkpoint the prompt continues from (else its end)
            // (not with a layer split: a slot's state is spread over the GPUs, and copy_from_slot reads one session)
            if (o.prompt_cache > 0 && req_imgs.empty() && !multi_gpu && !split_same)
                for (int b = 0; b < (int) bs.size(); ++b) {
                    const BSlot& sl = bs[(size_t) b];
                    if (sl.active || !sl.cached || sl.cvec != want_cvec) continue;
                    if ((int64_t) sl.ids.size() > std::max(resume, slot_tokens) && starts_with(sl.ids, {})) {
                        slot_source = b;
                        slot_tokens = (int64_t) sl.ids.size();
                        slot_ck = nullptr;
                    }
                    for (const ConvCheckpoint& c : sl.checks)
                        if ((int64_t) c.ids.size() > std::max(resume, slot_tokens) && starts_with(c.ids, c.imgs)) {
                            slot_source = b;
                            slot_tokens = (int64_t) c.ids.size();
                            slot_ck = &c;
                        }
                }
            const auto parked = conversations.best(ids, req_imgs, want_cvec);
            std::optional<strata::core::SavedConversation> incoming;
            int64_t in_tokens = 0;
            bool in_live = false;
            const char* in_source = "ram";
            // --conversation-save: a file goes further than the live session and every parked conversation
            const auto saved = disk.enabled() ? disk.best(ids, req_imgs, want_cvec)
                                              : strata::core::ConversationDisk::Match{};
            if (saved.tokens > std::max(resume, slot_tokens) && saved.tokens > parked.tokens) {
                const uint64_t floor = (uint64_t) o.conversation_cache_min_free_mib * 1024 * 1024;
                const size_t need = disk.load_bytes(saved.index);
                if (!strata::core::conversation_memory_admit(strata::core::conversation_available_memory(), need,
                                                             floor)) {
                    std::fprintf(stderr, "strata serve: conversation save: skip reading %zu MiB (physical RAM "
                                         "admission)\n", need >> 20);
                } else {
                    const auto t0 = Clock::now();
                    std::string de;
                    try {
                        strata::core::SavedConversation image;
                        if (disk.load(saved.index, image, de)) {
                            disk.touch(saved.index);
                            incoming.emplace(std::move(image));
                            in_tokens = saved.tokens;
                            in_live = saved.live;
                            in_source = "disk";
                            std::fprintf(stderr, "strata serve: conversation save: read %zu MiB in %.1f ms\n",
                                         need >> 20,
                                         std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                        } else {
                            std::fprintf(stderr, "strata serve: conversation save: %s\n", de.c_str());
                        }
                    } catch (const std::bad_alloc&) {
                        std::fprintf(stderr, "strata serve: conversation save: allocation failed; reading the prompt\n");
                    }
                }
            }
            if (!incoming && parked.tokens > std::max(resume, slot_tokens)) {
                incoming.emplace(conversations.take(parked.index));
                in_tokens = parked.tokens;
                in_live = parked.live;
            }
            // the whole image is checked before the outgoing state is parked or overwritten: an invalid one is
            // dropped and the request falls back to the prefix it had
            if (incoming && !snapshot_validate(*incoming, err)) {
                std::fprintf(stderr, "strata serve: conversation cache: discard invalid snapshot from %s (%s)\n",
                             in_source, err.c_str());
                incoming.reset();
                err.clear();
            }
            // the outgoing conversation is parked before a checkpoint rewind, a reset or the incoming restore
            // overwrites the positional state it needs
            if (incoming) slot_source = -1;
            if ((!from_live || incoming || slot_source >= 0) && !park_current(incoming ? incoming->bytes() : 0)) {
                std::printf("ERR %s\n", err.c_str());
                return 1;
            }
            if (slot_source >= 0) {
                const auto t0 = Clock::now();
                if (!copy_from_slot(slot_source, slot_ck, err)) {
                    // the main session may be half written: read this prompt from the start
                    std::fprintf(stderr, "strata batch: restoring slot %d failed (%s); reading the prompt\n",
                                 slot_source, err.c_str());
                    err.clear();
                    live.clear();
                    live_imgs.clear();
                    checks.clear();
                    resume = 0;
                    from_live = false;
                } else {
                    checks.clear();   // the main session's checkpoints were of the conversation it held before
                    // a read that gave way, the same request again (into its own slot): on with it - its parts and
                    // checkpoints as if it had not stopped
                    if (slot_ck == nullptr && bs[(size_t) slot_source].partial && admit_slot == slot_source) {
                        resumed_from0 = bs[(size_t) slot_source].partial_from0;
                        for (const ConvCheckpoint& c : bs[(size_t) slot_source].checks) checks.push_back(c);
                        bs[(size_t) slot_source].partial = false;
                    }
                    if (slot_ck != nullptr) {
                        live = slot_ck->ids;
                        checks.push_back(*slot_ck);   // the main session's chain has it now
                        checks.back().used = ++check_clock;
                    } else {
                        live = bs[(size_t) slot_source].ids;
                    }
                    live_imgs.clear();
                    resume = slot_tokens;
                    from_live = true;
                    std::fprintf(stderr, "strata batch: slot %d gave back %lld tokens of this conversation (%s) in "
                                 "%.1f ms\n", slot_source, (long long) slot_tokens,
                                 slot_ck != nullptr ? "its turn checkpoint" : "all it holds",
                                 std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                }
            }
            if (incoming) {
                const auto t0 = Clock::now();
                if ((conv_stages.empty()
                         ? strata::core::conversation_snapshot_restore(*incoming, ss, g, mtp.kv_state(), err)
                         : strata::core::conversation_split_restore(*incoming, conv_stages, g, mtp.kv_state(), err)) !=
                    strata::core::ConversationRestore::restored) {
                    // validated above: a failure here is fatal, never a session to decode from half restored
                    std::printf("ERR restoring parked conversation: %s\n", err.c_str());
                    return 1;
                }
                if (std::getenv("STRATA_SNAPSHOT_VERIFY") != nullptr) {
                    const strata::core::OnDevice on_draft(conv_stages.empty() ? -1 : conv_stages.back().device);
                    uint64_t draft_hash = 0;
                    if (!strata::core::conversation_kv_verify(incoming->kv.back(), mtp.kv_state(), g,
                                                              int64_t(incoming->live.ids.size()), false, draft_hash, err)) {
                        std::printf("ERR verifying restored draft KV: %s\n", err.c_str());
                        return 1;
                    }
                    std::fprintf(stderr, "strata serve: SNAPSHOT_VERIFY draft=%016llx cells=%lld mode=%d source=%s "
                                         "resident=%lld\n",
                                 (unsigned long long) draft_hash, (long long) incoming->kv.back().cells,
                                 mtp.kv_state().kv_mode, in_source,
                                 (long long) mtp.kv_state().n_slots * strata::kernels::qsa_real_shapes().page_size);
                }
                live = std::move(incoming->live.ids);
                live_imgs = std::move(incoming->live.imgs);
                checks = std::move(incoming->checkpoints);
                cvec_cached = incoming->cvec;
                resume = in_tokens;
                from_live = in_live;
                // (a layer split captures in full: its K/V is copied from several devices)
                if (std::getenv("STRATA_SNAPSHOT_FULL_CAPTURE") == nullptr && conv_stages.empty())
                    conversations.retain(std::move(incoming->kv), int64_t(live.size()));
                incoming.reset();   // the running state and checkpoint copies are not needed any more
                std::fprintf(stderr, "strata serve: conversation cache: restored %lld tokens (%s, from %s) in %.1f ms; "
                                     "parked=%zu bytes=%zu\n",
                             (long long) resume, from_live ? "live" : "checkpoint", in_source,
                             std::chrono::duration<double, std::milli>(Clock::now() - t0).count(),
                             conversations.size(), conversations.bytes());
            }
            if (want_cvec != cvec_cached) {
                live_ok = false;
                checks.clear();
                cvec_cached = want_cvec;
            }
            if (strata::kernels::cvec().loaded()) strata::kernels::cvec_set_enabled(want_cvec);
            // this request rewrites every cell from `resume` on, so a checkpoint past it (or not on this prompt's
            // path) no longer has its cells; the ones kept are prefixes of both the old tokens and the new
            checks.erase(std::remove_if(checks.begin(), checks.end(), [&](const ConvCheckpoint& c) {
                             return (int64_t) c.ids.size() > resume || !starts_with(c.ids, c.imgs);
                         }), checks.end());
            live_ok = false;   // until this request has finished, the session is in between
            int64_t reread_to = -1;   // STRATA_CKPT_REREAD only: read [0, reread_to) again instead of restoring
            if (resume == 0) {
                strata::core::session_zero(ss, g, nullptr, main_cs);
                strata::gpu::stream_sync(main_stream);
                for (auto& st : stages) {
                    const strata::core::OnDevice on(st->dev);
                    strata::core::session_zero(st->ss, g, nullptr, (void*) st->stream);
                    strata::gpu::stream_sync(st->stream);
                }
                checks.clear();
            } else if (!from_live) {
                ConvCheckpoint* c = nullptr;
                for (ConvCheckpoint& k : checks) if ((int64_t) k.ids.size() == resume) c = &k;
                if (c != nullptr) c->used = ++check_clock;   // mounting through it is the use LRU counts
                static const bool reread = std::getenv("STRATA_CKPT_REREAD") != nullptr;
                if (reread && c != nullptr) {
                    // THE CHECK OF THE CHECKPOINT: instead of restoring it, read its tokens again from position 0 in
                    // one run (below, with the prompt path's slots lent like any read) - the same chunks the request
                    // that saved it read them in, when that request started at 0.  With the VRAM expert set fixed
                    // (--adapt-swaps 0) the answer must match the restored one token for token; anything the
                    // checkpoint missed shows up as a difference.
                    strata::core::session_zero(ss, g, nullptr, main_cs);
                    strata::gpu::stream_sync(main_stream);
                    for (auto& st : stages) {
                        const strata::core::OnDevice on(st->dev);
                        strata::core::session_zero(st->ss, g, nullptr, (void*) st->stream);
                        strata::gpu::stream_sync(st->stream);
                    }
                    reread_to = resume;
                    std::fprintf(stderr, "strata serve: STRATA_CKPT_REREAD: reading %lld tokens again instead of "
                                         "restoring\n", (long long) resume);
                } else if (c == nullptr || !checkpoint_restore(*c, ss, g) || c->stage_parts.size() != stages.size() ||
                           [&] {
                               for (size_t i = 0; i < stages.size(); ++i) {
                                   const strata::core::OnDevice on(stages[i]->dev);
                                   if (!checkpoint_restore(c->stage_parts[i], stages[i]->ss, g)) return true;
                               }
                               return false;
                           }()) {
                    std::printf("ERR restoring a conversation checkpoint failed\n");
                    return 1;
                }
            }
            // KV streaming: the drafter's ring may hold cells past `resume` from a longer turn; the main layers'
            // host copies and slots are always current (every writer writes both), so they need nothing
            if (use_mtp && resume > 0 && reread_to <= 0) mtp.kv_restore(resume);
            tr("request", n, geni ? 1 : 0);
            if (use_mtp) mtp.set_prompt_len(n);
            const int64_t read_from = reread_to > 0 ? 0 : resume;
            conversations.limit_reuse(read_from);
            pp_total = n;
            pp_from = read_from;
            pp_reached = read_from;
            pp_t0 = r0;
            pp_next_check = reread_to > 0 || !req_ckpt ? INT64_MAX : resume + o.prompt_cache_every;   // ckpt=0: none
            pp_tail_saved = reread_to > 0 || !req_ckpt;   // ckpt=0 (#861): no tail checkpoint either
            {
                std::lock_guard<std::mutex> lk(part_mu);
                part_at.clear();
                std::fill(part_next.begin(), part_next.end(), pp_next_check);
            }
            std::printf("RESUME %lld\n", (long long) resume);   // before reading: this many prompt tokens are reused
            strata::core::progress_at("reading the prompt, from token", read_from);
            std::fflush(stdout);
            // A SHORT PART OF THE PROMPT - the new message of a chat that continues from a checkpoint, the assistant
            // header - goes through the verify windows, S tokens at a time, as decode reads them.  The batched path
            // costs ~300 ms per run however few tokens it has (it streams every expert the chunk routes to that is
            // not in VRAM over PCIe), and it borrows slots it must refill after (~180 ms); a window costs ~16 ms a
            // token, with the misses on the CPU.  Each part below is decided on its own, so a long first message is
            // read batched and its header still goes through the windows.  Picture rows need the batched path.
            // STRATA_CKPT_REREAD compares a restored checkpoint with a batched re-read, so it keeps every read batched.
            static const bool no_short = std::getenv("STRATA_CKPT_REREAD") != nullptr;
            auto windows_ok = [&](int64_t a, int64_t b) -> bool {
                if (no_short || b - a > o.short_read) return false;
                if (sp.embd_rows != nullptr)
                    for (int64_t i = a; i < b; ++i)
                        if (sp.embd_rows[i] != nullptr) return false;
                return true;
            };
            // tokens [a, b) through the windows: commit all of them, then give the draft layer their residuals
            auto read_windows = [&](int64_t a, int64_t b, std::string& e) -> bool {
                // every token is committed and the picks are discarded: no head sampling (see set_head_sampling)
                struct NoHeadSampling {
                    strata::core::Verifier& v;
                    explicit NoHeadSampling(strata::core::Verifier& x) : v(x) { v.set_head_sampling(false); }
                    ~NoHeadSampling() { v.set_head_sampling(true); }
                } no_head_sampling(ver);
                std::vector<int32_t> win((size_t) S), outw((size_t) S), nxt((size_t) S);
                for (int64_t q = a; q < b;) {
                    if (stop_req.load()) { e = "cancelled"; return false; }
                    const int T = (int) std::min<int64_t>(S, b - q);
                    for (int t = 0; t < T; ++t) {
                        win[(size_t) t] = (int32_t) cur[(size_t) (q + t)];
                        nxt[(size_t) t] = (int32_t) cur[(size_t) (q + t + 1)];
                    }
                    drive.d.layers = 0;
                    drive.d.experts = 0;
                    drive.d.failed = false;
                    if (!ver.run(T, win.data(), q, win_pool_fn, win_pool_user, outw.data(), e) || drive.d.failed) {
                        if (drive.d.failed && drive.d.fail) e = drive.d.fail;
                        return false;
                    }
                    // STRATA_LOGPOS=<path>: the teacher-forced log-probability of every token read here (every token
                    // is committed and nxt[t] is the prompt's own next token), appended to <path>; the last column
                    // is that of STRATA_LOGPOS_EXTRA (default 248046, <|im_end|>), so the end-of-turn mass a chat
                    // model puts on raw text can be taken out (upstream 06b82fe)
                    static std::FILE* logpos = [] {
                        const char* p = std::getenv("STRATA_LOGPOS");
                        return p != nullptr ? std::fopen(p, "ab") : nullptr;
                    }();
                    static const int32_t logpos_extra = [] {
                        const char* p = std::getenv("STRATA_LOGPOS_EXTRA");
                        return p != nullptr ? (int32_t) std::strtol(p, nullptr, 10) : (int32_t) 248046;
                    }();
                    if (logpos != nullptr && !ver.window_logprobs(nxt.data(), T, q, logpos_extra, logpos, e))
                        return false;
                    if (!ver.commit(T, e) || (use_mtp && !mtp.prefill(ver.final_R_all(), nxt.data(), T, q, e)))
                        return false;
                    q += T;
                    pp_reached = q;   // #471
                }
                // the batched prompt path (other queues), checkpoints and snapshots may follow: the last commit first
                if (!ver.wait_commit(e)) return false;
                const double ms = std::chrono::duration<double, std::milli>(Clock::now() - pp_t0).count();
                std::printf("PP %lld %lld %.0f %.1f\n", (long long) b, (long long) pp_total, ms,
                            ms > 0.0 ? 1000.0 * (double) (b - pp_from) / ms : 0.0);
                strata::core::progress_beat();
                std::fflush(stdout);
                return true;
            };
            // the batched path's slots are lent just before its first run and given back (refilled) before a window
            // reads - so the windows always see the whole expert cache - or once the prompt is read
            std::vector<std::pair<int32_t, int32_t>> lent_now;
            int64_t lent_chunk = 0;   // the chunk the lent slots hold the prompt path's buffers for
            // a residency index's GPU (a layer split: the stage that runs its layer; -1: CUDA0) and its cache
            auto stage_at = [&](int32_t i) -> int {
                int st = -1;
                while (st + 1 < (int) stages.size() && i / g.n_expert >= stages[(size_t) st + 1]->lb) ++st;
                return st;
            };
            auto refill = [&](std::string& e) -> bool {
                if (lent_now.empty()) return true;
                tr("refill start", (long long) lent_now.size());
                // each GPU's slots on its own queue (lent_now runs CUDA0's first, then each stage's), one wait each
                for (int st = -1; st < (int) stages.size(); ++st) {
                    strata::core::ExpertCache& c = st < 0 ? xcache : stages[(size_t) st]->cache;
                    const strata::core::OnDevice on(st < 0 ? -1 : stages[(size_t) st]->dev);
                    bool any = false;
                    for (const auto& [i, slot] : lent_now) {   // D-4: queued, one wait (STRATA_REFILL_BLOCKING=1: each)
                        if (stage_at(i) != st) continue;
                        // a transient source's blob is copied into fill_tmp, so each fill must be over before the next
                        const bool tb = srcp->transient(i / g.n_expert, i % g.n_expert);
                        const uint8_t* b = blob_to_copy(srcp, i / g.n_expert, i % g.n_expert, fill_tmp);
                        const int64_t nb = (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(i / g.n_expert);
                        if (b == nullptr || !(refill_blocking() || tb ? c.fill_slot_blocking(slot, b, e, nb)
                                                                      : c.fill_slot_queued(slot, b, e, nb)))
                            return false;
                        host_res[(size_t) i] = slot;
                        any = true;
                    }
                    if (any && !c.sync_queued(e)) return false;
                }
                res_upload();
                lent_now.clear();
                lent_chunk = 0;
                return true;
            };
            // lend the slots `tokens` batched prompt tokens need: the prompt path's buffers for min(chunk, tokens
            // rounded up to 256), laid out in the last of the slots it may borrow
            auto lend = [&](int64_t tokens, std::string& e) -> bool {
                if (lend_first < 0) return true;                       // its own buffers: nothing to lend
                const int64_t want = std::min<int64_t>(o.prefill_chunk, (tokens + 255) / 256 * 256);
                if (!lent_now.empty()) {
                    if (want <= lent_chunk) return true;
                    if (!refill(e)) return false;
                }
                const int32_t first = std::max<int32_t>(lend_first, (int32_t) (xcache.slots() - lend_slots(want)));
                if (want != sp.chunk() || first != lend_first_now) {
                    if (!sp.relayout(want, xcache.device_slot(first), lend_bytes(first), e)) return false;
                    lend_first_now = first;
                }
                // CUDA0's layers, then each later stage's from its own cache (a slot number names a slot of the cache
                // of the GPU that runs the layer)
                const size_t end0 = stages.empty() ? host_res.size() : (size_t) (stages[0]->lb * g.n_expert);
                for (size_t i = 0; i < end0; ++i)
                    if (host_res[i] >= first) {
                        lent_now.emplace_back((int32_t) i, host_res[i]);
                        host_res[i] = strata::core::kNotResident;
                    }
                for (size_t j = 0; j < stages.size(); ++j) {
                    GpuStage& st = *stages[j];
                    const int32_t sf = std::max<int32_t>(
                        st_lend_first[j], (int32_t) (st.cache.slots() - lend_slots_of(st.cache, st.ss, st.dev, want)));
                    if (want != st.sp.chunk() || sf != st_lend_first_now[j]) {
                        const strata::core::OnDevice on(st.dev);
                        if (!st.sp.relayout(want, st.cache.device_slot(sf), lend_bytes_of(st.cache, sf), e)) return false;
                        st_lend_first_now[j] = sf;
                    }
                    for (size_t i = (size_t) (st.lb * g.n_expert); i < (size_t) (st.le * g.n_expert); ++i)
                        if (host_res[i] >= sf) {
                            lent_now.emplace_back((int32_t) i, host_res[i]);
                            host_res[i] = strata::core::kNotResident;
                        }
                }
                res_upload();
                lent_chunk = want;
                return true;
            };
            // --batch: a prompt read while slots are decoding goes one prompt chunk at a time (the same chunks as one
            // run: each run starts where the last ended, at a multiple of the chunk), and between two chunks the slots
            // decode for a share of the time the chunk took (STRATA_BATCH_DECODE_SHARE, default 0.5), so a long
            // prompt does not stop the others.  The slots' windows then see the cache without the slots this prompt
            // borrowed (marked missing: those experts run on the CPU), never a prompt buffer (#465).
            static const double decode_share = [] {
                const char* v = std::getenv("STRATA_BATCH_DECODE_SHARE");
                return v != nullptr ? std::max(0.0, std::strtod(v, nullptr)) : 0.5;
            }();
            bool batch_fatal = false;
            int64_t il_parts = 0;   // the slots' windows run between this prompt's chunks
            double il_ms = 0;
            int64_t yielded_at = -1;   // BYIELD: the read stopped here and its slot holds it
            int yielded_slot = -1;
            // `BYIELD <slot>` (the server: a shorter request is waiting): at the next chunk boundary the part read so
            // far is copied into <slot> (this admission's own, or a free one the server reserved for a solo request),
            // the request ends with `YIELDED <slot> <tokens>` + DONE cancel, and the server sends it again later: it
            // continues from the slot with the same chunks (upstream #656's cooperative preemption, a slot as the park)
            auto read_part = [&](int64_t a0, int64_t b0, std::string& e) -> bool {
                if (o.batch <= 0) return sp.run(ids.data() + a0, b0 - a0, a0, e);
                const int64_t C = std::max<int64_t>(sp.chunk(), 1);
                for (int64_t q = a0; q < b0;) {
                    const int64_t r = std::min(b0, q + C);
                    const auto tq = Clock::now();
                    if (!sp.run(ids.data() + q, r - q, q, e)) return false;
                    q = r;
                    if (q >= b0) continue;
                    int ys = -1;
                    {   // a BSTOP that came meanwhile ends its slot at its next window; a BYIELD is for this read
                        std::lock_guard<std::mutex> lk(in_mu);
                        for (auto it = in_lines.begin(); it != in_lines.end();) {
                            if (it->rfind("BSTOP ", 0) == 0) {
                                const int b = (int) std::strtol(it->c_str() + 6, nullptr, 10);
                                if (b >= 0 && b < (int) bs.size()) bs[(size_t) b].stop = true;
                                it = in_lines.erase(it);
                            } else if (it->rfind("BYIELD ", 0) == 0) {
                                ys = (int) std::strtol(it->c_str() + 7, nullptr, 10);
                                it = in_lines.erase(it);
                            } else {
                                ++it;
                            }
                        }
                    }
                    if (ys >= 0) {
                        // only where the rest is read the same way after it (more than a chunk and more than a short
                        // read left), text only, and into a slot that is not decoding
                        const bool can = ys < (int) bs.size() && !bs[(size_t) ys].active &&
                                         (admit_slot < 0 || ys == admit_slot) && req_imgs.empty() && o.prompt_cache > 0 &&
                                         b0 - q > std::max<int64_t>(C, o.short_read);
                        std::string ye;
                        const auto ty = Clock::now();
                        std::vector<int32_t> pre(ids.begin(), ids.begin() + q);
                        if (can && copy_to_slot(ys, pre, ye)) {
                            BSlot& sl = bs[(size_t) ys];
                            sl = BSlot{};
                            sl.ids = std::move(pre);
                            sl.cached = true;
                            sl.partial = true;
                            sl.partial_from0 = read_from == 0 || resumed_from0;
                            sl.cvec = cvec_cached;
                            for (const ConvCheckpoint& c : checks)   // the root / periodic checkpoints of this read
                                if ((int64_t) c.ids.size() <= q && std::equal(c.ids.begin(), c.ids.end(), ids.begin(),
                                        [](int32_t xa, int64_t yb) { return (int64_t) xa == yb; }))
                                    sl.checks.push_back(c);
                            yielded_at = q;
                            yielded_slot = ys;
                            std::fprintf(stderr, "strata batch: the prompt read gives way at %lld of %lld tokens; slot %d "
                                                 "holds it (copied in %.1f ms)\n", (long long) q, (long long) n, ys,
                                         std::chrono::duration<double, std::milli>(Clock::now() - ty).count());
                            e = "yield";
                            return false;
                        }
                        if (can) bs[(size_t) ys].cached = false;   // half written
                        std::fprintf(stderr, "strata batch: BYIELD %d not taken at %lld of %lld%s%s\n", ys, (long long) q,
                                     (long long) n, ye.empty() ? "" : ": ", ye.c_str());
                    }
                    if (decode_share <= 0.0 || !batch_on()) continue;
                    const double budget = decode_share * std::chrono::duration<double, std::milli>(Clock::now() - tq).count();
                    const auto td = Clock::now();
                    do {
                        if (!batch_step()) { batch_fatal = true; e = "a batch window failed"; return false; }
                    } while (batch_on() && std::chrono::duration<double, std::milli>(Clock::now() - td).count() < budget);
                    ++il_parts;
                    il_ms += std::chrono::duration<double, std::milli>(Clock::now() - td).count();
                    if (trace) {
                        std::fprintf(stderr, "strata trace: prompt chunk to %lld, then the slots decoded %.0f ms\n",
                                     (long long) q, std::chrono::duration<double, std::milli>(Clock::now() - td).count());
                        std::fflush(stderr);
                    }
                }
                return true;
            };
            apply_pending(true);
            // per-request sampling for the verify window's head (greedy when temperature is absent)
            strata::kernels::SamplerParams req_sp;
            req_sp.greedy = req_temperature <= 0.0f;
            req_sp.temperature = req_temperature;
            req_sp.top_p = req_top_p;
            req_sp.top_k = req_top_k;
            req_sp.seed = req_seed ? req_seed
                                   : (unsigned long long) std::chrono::steady_clock::now().time_since_epoch().count();
            req_sp.min_p = std::clamp(req_min_p, 0.0f, 1.0f);
            req_sp.penalty_last_n = std::max(req_penalty_last_n, 0);
            req_sp.penalty_repeat = req_penalty_repeat;
            req_sp.penalty_freq = req_penalty_freq;
            req_sp.penalty_present = req_penalty_present;
            req_sp.counter = 0;
            ver.set_sampling(req_sp);
            if (use_mtp) mtp.set_draft_sampling(req_sp);
            drive.d.pcie_num = std::max(0, std::min(256, (int) (req_pcie_frac * 256.0 + 0.5)));
            // a layer split: CUDA0's share as asked; a later GPU keeps its own (its link) unless the request sets one
            for (int st = 0; st < split_drive.n; ++st)
                split_drive.pcie_num[st] = (st == 0 || split_same || req_pcie_frac != o.pcie_frac)
                                               ? drive.d.pcie_num : pcie_num_of(stages[(size_t) st - 1]->pcie_frac);
            const int hist_n = std::min(req_sp.penalty_last_n, kPenaltyWindowCap);
            ver.set_history(hist_n > 0 ? d_hist : nullptr, hist_n);
            bool cancelled = false;
            tr("prompt start", n - 1);
            // The prompt is read in two parts when it has a turn boundary past `resume`: up to the last <|im_start|>
            // (the conversation so far), a checkpoint there, then the new turn's header.  The next request of the same
            // chat renders the same history - but not always the same header or the thinking of this reply - so that
            // checkpoint is the one it reuses.
            int64_t turn_at = -1;
            if (o.prompt_cache > 0 && o.turn_token >= 0)
                for (int64_t i = n - 1; i > resume; --i)
                    if (ids[(size_t) i] == o.turn_token) { turn_at = i; break; }
            // #458 (opt-in): a short turn of --tail-role-token's role right before the new assistant turn (the
            // server's trailing reasoning-effort turn) stays out of the checkpoint, so the next request - another
            // effort, or the next turn of the chat - finds the conversation without it
            if (turn_at > 0 && o.tail_role_token >= 0)
                for (int64_t i = turn_at - 1; i > resume; --i)
                    if (ids[(size_t) i] == o.turn_token) {
                        if (ids[(size_t) i + 1] == o.tail_role_token) turn_at = i;
                        break;
                    }
            // A prompt read from token 0 also stops at its FIRST turn boundary: the end of the system prompt (with
            // the tools), which every new chat of the same client shares.  That checkpoint becomes the chain's root,
            // which the retention policy pins (conv_cache.hpp), so the next new chat reads only what comes after it.
            // (PR #65, code-martin.)  Only for a system prompt of --prompt-cache-root tokens or more: a small one
            // is cheaper to read again than the extra part costs (~0.3 s).
            const int64_t last_turn = turn_at;   // ckpt=0 drops the split there, the root is still looked for before it
            if (!req_ckpt) turn_at = -1;
            int64_t root_at = -1;
            if (o.prompt_cache > 0 && o.turn_token >= 0 && o.prompt_cache_root > 0 && (read_from == 0 || resumed_from0))
                for (int64_t i = 1; i < last_turn; ++i)
                    if (ids[(size_t) i] == o.turn_token) {
                        if (i >= o.prompt_cache_root) root_at = i;
                        break;
                    }
            static const bool message_checkpoint = [] {
                const char* e = std::getenv("STRATA_CACHE_MESSAGE_BOUNDARY");
                return e != nullptr && std::strtol(e, nullptr, 10) != 0;
            }();
            // Only add a snapshot; the existing token and image checks still decide reuse.
            const int64_t message_at = message_checkpoint && !multi_gpu && o.prompt_cache > 0
                ? strata::program::message_checkpoint_boundary(ids, resume, turn_at, o.turn_token) : -1;
            int64_t at = read_from;
            for (const int64_t to : {reread_to, root_at, message_at, turn_at, n - 1}) {
                if (to <= at) continue;
                err.clear();
                const bool win = windows_ok(at, to);
                if (win && !refill(err)) {
                    std::printf("ERR refilling a lent slot failed: %s\n", err.c_str());
                    return 1;
                }
                if (!win && !lend(to - at, err)) {
                    std::printf("ERR lending the prompt path its slots failed: %s\n", err.c_str());
                    return 1;
                }
                const auto tsp = Clock::now();
                const bool sp_ok = win ? read_windows(at, to, err) : read_part(at, to, err);
                if (trace) {
                    std::fprintf(stderr, "strata trace: read %lld tokens (%s) in %.1f ms\n", (long long) (to - at),
                                 win ? "windows" : "batched",
                                 std::chrono::duration<double, std::milli>(Clock::now() - tsp).count());
                    std::fflush(stderr);
                }
                if (!sp_ok) {
                    if (yielded_at < 0 && (batch_fatal || !stop_req.load())) {
                        std::fprintf(stderr, "strata serve: %s\n", err.c_str());
                        std::printf("ERR %s\n", err.c_str());
                        return 1;
                    }
                    cancelled = true;   // stopped while reading the prompt: refill the lent slots below, then DONE cancel
                    break;
                }
                at = to;
                if ((to == turn_at || to == root_at || to == message_at) &&
                    !checkpoint_at(to, nullptr, to == message_at && to != turn_at && to != root_at)) {   // the tail kind leaves first
                    std::printf("ERR saving a conversation checkpoint failed\n");
                    return 1;
                }
                if (trace && to == message_at)
                    std::fprintf(stderr, "strata serve: message boundary checkpoint: %lld tokens, %lld tail\n",
                                 (long long) message_at, (long long) (turn_at - message_at));
            }
            if (!refill(err)) {
                std::printf("ERR refilling a lent slot failed: %s\n", err.c_str());
                return 1;
            }
            tr("prompt done (slots refilled)");
            if (il_parts > 0)
                std::fprintf(stderr, "strata batch: the prompt was read in %lld parts, the slots decoding %.0f ms between "
                                     "them\n", (long long) il_parts + 1, il_ms);
            const double prompt_ms = std::chrono::duration<double, std::milli>(Clock::now() - r0).count();
            std::printf("REUSED %lld\n", (long long) resume);   // the prompt is read; the first window comes next
            std::fflush(stdout);
            // the verify windows: the first holds the last prompt token alone
            int64_t p = n - 1;
            int32_t x = (int32_t) ids[(size_t) (n - 1)];
            std::vector<int32_t> drafts((size_t) S, 0), window((size_t) S), outv((size_t) S);
            std::vector<float> dprob((size_t) S, 0.0f);
            std::vector<int32_t> sbuf((size_t) S, 0);
            const bool sfx_on = o.suffix_draft > 0 || o.lookup_chain > 0;
            if (sfx_on) {
                sfx.reset();
                for (int64_t t : ids) sfx.append((int32_t) t);
            }
            if (o.lookup_chain > 0) {
                extra_sources_reset();
                for (int64_t t : ids) { const int32_t t32 = (int32_t) t; extra_sources_append(&t32, 1); }
            }
            bool first_window = true;
            int64_t produced_n = 0, sfx_windows = 0, sfx_drafts = 0, sfx_ok = 0;
            int64_t chain_windows = 0, chain_drafts = 0, chain_ok = 0;   // --lookup-chain's own counts
            int64_t t2_rej[8] = {}, t2_hit[8] = {};   // STRATA_MTP_TOP2: rejections inside the MTP drafts by depth, runner-up hits
            std::vector<int32_t> cbuf((size_t) S, 0), ctail;
            strata::spec::PromptLookupSource lookup_src(sfx);
            static const bool chain_fixed = std::getenv("STRATA_LOOKUP_CHAIN_FIXED") != nullptr;   // no policy gate
            int64_t draft_offered = 0, draft_accepted = 0;
            // what the session holds once this request is done: the prompt read so far, then every committed token
            std::vector<int32_t> consumed;
            consumed.reserve((size_t) (n + max_new + S));
            for (int64_t i = 0; i < n - 1; ++i) consumed.push_back((int32_t) ids[(size_t) i]);
            const char* finish = "length";
            const Clock::time_point d0 = Clock::now();
            // STRATA_DECODE_TIMING=1: where a request's decode time goes (one line per request)
            static const bool dec_timing = std::getenv("STRATA_DECODE_TIMING") != nullptr;
            struct DecSnap {
                double wait, pool, host, plan, actq, jobs, run;
                int64_t misses, entries, hits, pcie;
            };
            auto dec_snap = [&]() {
                return DecSnap{ver.ms_wait, ver.ms_pool, ver.ms_host, drive.d.ms_plan, drive.d.ms_actq, drive.d.ms_jobs,
                               drive.d.ms_run, drive.d.multi_misses, drive.d.multi_entries, drive.d.cache_hits,
                               drive.d.pcie_experts};
            };
            const DecSnap ds0 = dec_snap();
            double dt_run = 0, dt_commit = 0, dt_draft = 0, dt_pre = 0, dt_join = 0, dt_adapt = 0;
            const int64_t adapt_swaps0 = adapt_swaps_n, adapt_sync0 = adapt_sync_n;
            int64_t dec_windows = 0, dec_T = 0;
            const int64_t decode_hits0 = drive.d.cache_hits;
            // the expert tiers this request read from (DONE's RAM / file blobs and file MB)
            const int64_t ram0 = src.ram_reads(), files0 = src.file_reads();
            const uint64_t file_bytes0 = src.file_read_bytes();
            const int64_t decode_look0 = drive.d.cache_hits + drive.d.cache_admitted + drive.d.cache_refused;
            const int64_t offload0 = drive.d.offload_entries;   // #588
            if (cancelled) finish = "cancel";
            while (!cancelled && produced_n < max_new) {
                int T = use_mtp ? S_mtp : 1;   // no --mtp: one token a round unless a lookup draft fires
                if (use_mtp && req_spec_min_p > 0.0) {
                    T = 1;
                    while (T < S_mtp && dprob[(size_t) T - 1] >= (float) req_spec_min_p) ++T;
                }
                if (first_window) T = 1;
                // a repeat of earlier context (prompt lookup) where the MTP's own first guess agrees: the policy takes it
                // when its expected tokens per ms, from the measured acceptance and window costs, beat the MTP window's
                bool from_sfx = false;
                int sfx_match = 0;
                if (o.suffix_draft > 0 && !first_window) {
                    const int k = sfx.propose(S - 1, sbuf.data());
                    sfx_match = sfx.last_match();
                    if (k > 0 && (!use_mtp || sbuf[0] == drafts[0])) {
                        const strata::spec::DraftPolicy::Pick pk = policy.choose(T, k, sfx_match);
                        if (pk.lookup) { T = pk.t; from_sfx = true; }
                    }
                }
                // --lookup-chain: what followed an earlier occurrence of the context + the MTP's drafts, after them
                int chain_n = 0, cm = 0;
                if (o.lookup_chain > 0 && !first_window && !from_sfx && T < S) {
                    int csrc = -1;
                    const int nt = chain_tail(sfx, drafts.data(), T - 1, ctail);
                    chain_n = strata::spec::propose_from_sources(lookup_src, ctail.data(), nt, T - 1,
                                                                 std::min(o.lookup_chain, S - T), o.lookup_chain_min,
                                                                 cbuf.data(), &cm, &csrc);
                    // the policy keeps the chained tokens that pay at this machine's measured window costs
                    if (chain_n > 0 && !chain_fixed) {
                        double p_mtp = 1.0;
                        for (int i = 0; i < T - 1; ++i) p_mtp *= dprob[(size_t) i];
                        chain_n = policy.chain(T, p_mtp, chain_n, cm);
                    }
                }
                const int T_mtp = T;
                T += chain_n;
                const bool timed_round = !first_window;
                const Clock::time_point round0 = Clock::now();
                if (p + T > o.max_context) break;
                window[0] = x;
                for (int i = 1; i < T_mtp; ++i) window[(size_t) i] = from_sfx ? sbuf[(size_t) i - 1] : drafts[(size_t) i - 1];
                for (int i = 0; i < chain_n; ++i) window[(size_t) T_mtp + (size_t) i] = cbuf[(size_t) i];
                drive.d.layers = 0;
                drive.d.experts = 0;
                drive.d.failed = false;
                // #463: the previous adapt round's copies land first - with a non-blocking query, whether a swapped-in
                // expert ran on the GPU or the CPU (they round differently) depended on the copy's timing
                // (STRATA_ADAPT_NOWAIT=1: 0.1.37's non-blocking query, the A/B)
                apply_pending(!adapt_nowait());
                if (hist_n > 0) {
                    // the tails the penalties count over, ONE PER ROW: the tokens the state has consumed, the
                    // fed-back head `x` (it joins `consumed` only after this window commits), then the drafts
                    // before that row - what plain decode would have counted there.  (Until 0.1.19 only row 0
                    // was staged, and the drafted rows read unwritten slots.)
                    strata::kernels::penalty_rows(consumed.data(), (int64_t) consumed.size(), window.data(), T,
                                                  hist_n, hist_stage.data());
                    const strata::core::OnDevice on_h(hist_dev);
                    strata::gpu::copy(d_hist, hist_stage.data(), (size_t) T * (size_t) hist_n * sizeof(int32_t));
                }
                tr("window", p, T);
                const Clock::time_point tw0 = Clock::now();
                dt_pre += std::chrono::duration<double, std::milli>(tw0 - round0).count();
                if (!ver.run(T, window.data(), p, win_pool_fn, win_pool_user, outv.data(), err) || drive.d.failed) {
                    std::printf("ERR %s\n", drive.d.failed && drive.d.fail ? drive.d.fail : err.c_str());
                    return 1;
                }
                int a = 0;
                while (a < T - 1 && window[(size_t) a + 1] == outv[(size_t) a]) ++a;
                if (from_sfx) { ++sfx_windows; sfx_drafts += T - 1; sfx_ok += a; }
                if (chain_n > 0) { ++chain_windows; chain_drafts += chain_n; chain_ok += std::max(0, a - (T_mtp - 1)); }
                if (strata::core::MtpDrafter::top2_env() && !from_sfx && !first_window && a < T_mtp - 1 && a < 8) {
                    ++t2_rej[a];
                    if (mtp.top2(a) == outv[(size_t) a]) ++t2_hit[a];
                }
                const Clock::time_point tw1 = Clock::now();
                std::thread adapt_thr;   // the adaptive tier beside the commit and the draft (as in generate)
                bool adapt_ok = true;
                if (!drive.d.usage.empty() && ((rounds + 1) % o.adapt_every) == 0)
                    adapt_thr = std::thread([&] {
                        const Clock::time_point ta = Clock::now();
                        adapt_ok = adapt();
                        dt_adapt += std::chrono::duration<double, std::milli>(Clock::now() - ta).count();
                    });
                if (!ver.commit(a + 1, err)) {
                    if (adapt_thr.joinable()) adapt_thr.join();
                    std::printf("ERR %s\n", err.c_str());
                    return 1;
                }
                // the window's first a + 1 tokens are in the session now (the last output is not: it is next x)
                for (int i = 0; i <= a; ++i) consumed.push_back(window[(size_t) i]);
                draft_offered += T - 1;
                draft_accepted += a;
                first_window = false;
                bool eos = false;
                for (int i = 0; i <= a && produced_n < max_new && !eos; ++i) {
                    std::printf("T %d\n", (int) outv[(size_t) i]);
                    if (req_logprobs >= 0) {   // row i of this window's head logits is the distribution token i came from
                        static std::vector<float> lrow;
                        static std::vector<int32_t> lord;
                        lrow.resize((size_t) ver.vocab());
                        if (!ver.copy_logits(i, lrow.data())) {
                            std::printf("LP nan\n");
                        } else {
                            float mx = -INFINITY;
                            for (const float v : lrow) mx = std::max(mx, v);
                            double se = 0.0;
                            for (const float v : lrow) se += std::exp((double) v - mx);
                            const double lse = (double) mx + std::log(se);
                            std::printf("LP %.6f", (double) lrow[(size_t) outv[(size_t) i]] - lse);
                            if (req_logprobs > 0) {
                                lord.resize(lrow.size());
                                for (size_t v = 0; v < lrow.size(); ++v) lord[v] = (int32_t) v;
                                std::partial_sort(lord.begin(), lord.begin() + req_logprobs, lord.end(),
                                                  [&](int32_t x1, int32_t x2) { return lrow[(size_t) x1] > lrow[(size_t) x2]; });
                                for (int j = 0; j < req_logprobs; ++j)
                                    std::printf(" %d:%.6f", lord[(size_t) j], (double) lrow[(size_t) lord[(size_t) j]] - lse);
                            }
                            std::printf("\n");
                        }
                    }
                    strata::core::progress_beat();
                    ++produced_n;
                    if (sfx_on) sfx.append(outv[(size_t) i]);
                    if (o.lookup_chain > 0) extra_sources_append(&outv[(size_t) i], 1);
                    eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outv[(size_t) i]) != o.eos_ids.end();
                }
                std::fflush(stdout);
                ++rounds;
                const Clock::time_point tw2 = Clock::now();
                if (use_mtp && hist_n > 0 && mtp.coupled() && !eos && produced_n < max_new)
                    mtp.set_draft_history(consumed.data(), (int64_t) consumed.size(), outv[(size_t) a]);
                const bool drafted = !use_mtp || eos || produced_n >= max_new ||
                                     mtp.draft(T, outv.data(), p, a, drafts.data(), err, dprob.data(), (float) req_spec_min_p);
                {
                    const Clock::time_point tw3 = Clock::now();
                    auto msd = [](Clock::time_point a0, Clock::time_point b0) { return std::chrono::duration<double, std::milli>(b0 - a0).count(); };
                    dt_run += msd(tw0, tw1); dt_commit += msd(tw1, tw2); dt_draft += msd(tw2, tw3);
                    ++dec_windows; dec_T += T;
                }
                const Clock::time_point tj0 = Clock::now();
                if (adapt_thr.joinable()) adapt_thr.join();
                dt_join += std::chrono::duration<double, std::milli>(Clock::now() - tj0).count();
                if (!adapt_ok) {
                    std::printf("ERR an adaptive refill failed\n");
                    return 1;
                }
                if (!drafted) {
                    std::printf("ERR %s\n", err.c_str());
                    return 1;
                }
                if (timed_round && !eos && chain_n == 0)
                    policy.observe(from_sfx, T, a, sfx_match,
                                   std::chrono::duration<double, std::milli>(Clock::now() - round0).count());
                else if (timed_round && !eos)   // a chained window: its MTP part, its chain, its cost
                    policy.observe_chain(T_mtp, chain_n, a, cm,
                                         std::chrono::duration<double, std::milli>(Clock::now() - round0).count());
                if (eos) { finish = "stop"; break; }
                if (stop_req.load()) { finish = "cancel"; break; }
                x = outv[(size_t) a];
                p += a + 1;
            }
            const double decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - d0).count();
            // the last commit (set_commit_async): the session is complete before anything reads or copies it
            if (!ver.wait_commit(err)) {
                std::printf("ERR %s\n", err.c_str());
                return 1;
            }
            if (dec_timing && dec_windows > 0) {
                const DecSnap d1 = dec_snap();
                const double w = (double) dec_windows, L = (double) g.n_layers;
                std::fprintf(stderr, "strata decode timing: %lld windows, avg T %.2f, %.2f tokens/window, %.2f ms/window = "
                                     "verify %.2f (GPU-reach wait %.2f + per-layer host %.2f [plan %.2f actq %.2f jobs %.2f "
                                     "CPU %.2f] + stage %.2f) + commit/emit %.2f + draft %.2f; per layer-window: CPU experts "
                                     "%.2f (%.2f entries), VRAM hits %.2f, PCIe %.2f; before the window %.2f, adapt join %.2f "
                                     "(adapt %.2f, %lld swaps, %lld synchronous)\n",
                             (long long) dec_windows, dec_T / w, produced_n / w, decode_ms / w, dt_run / w,
                             (d1.wait - ds0.wait) / w, (d1.pool - ds0.pool) / w, (d1.plan - ds0.plan) / w,
                             (d1.actq - ds0.actq) / w, (d1.jobs - ds0.jobs) / w, (d1.run - ds0.run) / w,
                             (d1.host - ds0.host) / w, dt_commit / w, dt_draft / w, (d1.misses - ds0.misses) / (w * L),
                             (double) (d1.entries - ds0.entries) / (w * L), (double) (d1.hits - ds0.hits) / (w * L),
                             (double) (d1.pcie - ds0.pcie) / (w * L), dt_pre / w, dt_join / w, dt_adapt / w, (long long) (adapt_swaps_n - adapt_swaps0),
                             (long long) (adapt_sync_n - adapt_sync0));
                for (int st = 0; st < n_stages; ++st) {   // every stage's GPU profile (upstream fe9c10ca, b4405a0e)
                    const std::string pr = stage_ver(st).profile_report();
                    if (pr.empty()) continue;
                    if (st == 0)   // the first card's line keeps its text
                        std::fprintf(stderr, "strata decode GPU stages (millions of device-clock ticks per window):%s\n", pr.c_str());
                    else
                        std::fprintf(stderr, "strata decode GPU stages, stage %d (millions of device-clock ticks per window):%s\n",
                                     st, pr.c_str());
                }
            }
            if (!cancelled) {
                // a prompt stopped halfway leaves the session somewhere between two chunks: nothing to continue from
                // (the checkpoints taken while reading it are still good)
                live.swap(consumed);
                live_imgs = imgs_below(req_imgs, (int64_t) live.size());
                live_ok = o.prompt_cache > 0 && req_ckpt;   // ckpt=0: nothing to continue or park (#830)
            }
            static const bool state_hash = std::getenv("STRATA_STATE_HASH") != nullptr;
            if (state_hash && !cancelled && o.prompt_cache > 0) {   // every finished request, ckpt=0 too (parity gates)
                // DEBUG: a fingerprint of every part of the session over the positions it holds ([0, L)), and
                // separately of what lies past them in the last KV page (stale cells, fine unless something reads them)
                strata::gpu::device_sync();
                const int64_t L = (int64_t) live.size();
                const strata::kernels::QsaShapes qs = [&] {
                    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
                    s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_dim = g.idx_key_dim;
                    return s;
                }();
                auto hash_dev = [&](const void* p, size_t bytes, uint64_t h) {
                    std::vector<uint8_t> b(bytes);
                    if (bytes) strata::gpu::copy(b.data(), p, bytes);   // VRAM or a streamed host copy
                    return fnv1a(b.data(), b.size(), h);
                };
                // the cells [c0, c1) of one int8 K or V pool ([page][kv_head][page_size][head_dim]), bytes per value `w`
                auto hash_cells = [&](const void* pool, int64_t per_cell, int64_t c0, int64_t c1, uint64_t h) {
                    const int64_t ps = qs.page_size;
                    for (int64_t pg = c0 / ps; pg * ps < c1; ++pg)
                        for (int64_t hd = 0; hd < qs.n_head_kv; ++hd) {
                            const int64_t a = std::max(c0, pg * ps) - pg * ps, e = std::min(c1, (pg + 1) * ps) - pg * ps;
                            const size_t off = (size_t) (((pg * qs.n_head_kv + hd) * ps + a) * per_cell);
                            h = hash_dev((const uint8_t*) pool + off, (size_t) ((e - a) * per_cell), h);
                        }
                    return h;
                };
                const ConvStateSizes z = conv_state_sizes(g);
                // the GDN rows this session owns (a layer split's carve holds its own layers only)
                const size_t per_gdn = g.n_gdn_layers() > 0 ? z.gdn / (size_t) g.n_gdn_layers() : 0;
                uint64_t h_gdn = hash_dev(ss.gdn_state, per_gdn * (size_t) ss.gdn_alloc, 1469598103934665603ull);
                if (std::getenv("STRATA_STATE_HASH_GDN") != nullptr && ss.gdn_alloc > 0) {   // which layer first
                    const size_t per = per_gdn;
                    std::string s;
                    char b[8];
                    for (int64_t i = 0; i < ss.gdn_alloc; ++i) {
                        std::snprintf(b, sizeof(b), "%04llx ", (unsigned long long) (hash_dev((const uint8_t*) ss.gdn_state + i * per, per, 1469598103934665603ull) & 0xffff));
                        s += b;
                    }
                    std::fprintf(stderr, "strata serve: STATE_HASH_GDN %s\n", s.c_str());
                }
                uint64_t h_ple = hash_dev(ss.ple_hist, ss.ple_hist ? z.ple : 0, 1469598103934665603ull);
                uint64_t h_tail = 1469598103934665603ull, h_pool = h_tail, h_kv = h_tail, h_stale = h_tail;
                // pooled= the completed rows [0, L / idx_block) only; pooled_full= adds the spare row at L / idx_block
                // (the `dead` key the next block completion overwrites), which a conversation restore writes back
                uint64_t h_dead = h_tail, h_pool_full = h_tail;
                const int64_t kvb = qs.head_dim, scb = (qs.head_dim / 64) * 2;
                // a state's K/V arrays and their bytes per (cell, head) row: the host copy when it has one
                auto kv_arrays = [&](const strata::core::QsaState& st) {
                    const bool h = st.kv_mode != 0;
                    std::vector<std::pair<const void*, int64_t>> a;
                    if (st.kv_q4) {
                        const int64_t q4b = (int64_t) strata::kernels::kv_q4_bytes_per_head((int) qs.head_dim);
                        a = {{h ? st.host.k_q4 : st.k_q4, q4b}, {h ? st.host.v_q4 : st.v_q4, q4b}};
                    } else if (st.kv_hybrid) {
                        const int64_t q4b = (int64_t) strata::kernels::kv_q4_bytes_per_head((int) qs.head_dim);
                        a = {{h ? st.host.k_q : st.k_q, kvb}, {h ? st.host.v_q4 : st.v_q4, q4b},
                             {h ? st.host.k_scale : st.k_scale, scb}};
                    } else if (st.kv_int8) {
                        a = {{h ? st.host.k_q : st.k_q, kvb}, {h ? st.host.v_q : st.v_q, kvb},
                             {h ? st.host.k_scale : st.k_scale, scb}, {h ? st.host.v_scale : st.v_scale, scb}};
                    } else {
                        a = {{h ? st.host.k_pool : st.k_pool, qs.head_dim * 2},
                             {h ? st.host.v_pool : st.v_pool, qs.head_dim * 2}};
                    }
                    return a;
                };
                const int64_t end_cell = std::min<int64_t>(((L + qs.page_size - 1) / qs.page_size) * qs.page_size,
                                                           ss.max_cells);
                for (int64_t i = ss.qsa_ord0; i < ss.qsa_ord0 + ss.qsa_alloc; ++i) {   // the owned QSA layers
                    const strata::core::QsaState& st = ss.qsa_states[i];
                    h_tail = hash_dev(st.idx_tail, z.tail, h_tail);
                    h_dead = hash_dev(st.idx_dead, z.dead, h_dead);
                    h_pool = hash_dev(st.idx_pooled, (size_t) (L / qs.idx_block) * qs.idx_dim * 4, h_pool);
                    h_pool_full = hash_dev(st.idx_pooled, (size_t) (L > 0 ? L / qs.idx_block + 1 : 0) * qs.idx_dim * 4,
                                           h_pool_full);
                    // KV streaming: the host copy is the identity layout and holds every cell
                    for (const auto& [pool, w] : kv_arrays(st)) {
                        h_kv = hash_cells(pool, w, 0, L, h_kv);
                        h_stale = hash_cells(pool, w, L, end_cell, h_stale);
                    }
                }
                const strata::core::QsaState& ms = mtp.kv_state();
                uint64_t h_mtp = 1469598103934665603ull;
                const int64_t mL = std::min<int64_t>(L, ms.max_cells);
                for (const auto& [pool, w] : kv_arrays(ms))
                    if (pool != nullptr) h_mtp = hash_cells(pool, w, 0, mL, h_mtp);
                std::fprintf(stderr, "strata serve: STATE_HASH L=%lld gdn=%016llx ple=%016llx tail=%016llx pooled=%016llx "
                                     "kv=%016llx mtp=%016llx stale=%016llx dead=%016llx pooled_full=%016llx ple_prev=%d,%d\n",
                             (long long) L,
                             (unsigned long long) h_gdn, (unsigned long long) h_ple, (unsigned long long) h_tail,
                             (unsigned long long) h_pool, (unsigned long long) h_kv, (unsigned long long) h_mtp,
                             (unsigned long long) h_stale, (unsigned long long) h_dead,
                             (unsigned long long) h_pool_full, ss.ple_prev[0], ss.ple_prev[1]);
            }
            const int64_t req_hits = drive.d.cache_hits - decode_hits0;
            const int64_t req_look = (drive.d.cache_hits + drive.d.cache_admitted + drive.d.cache_refused) - decode_look0;
            const int64_t req_offload = drive.d.offload_entries - offload0;
            // #471: the prompt tokens this request read - all the fresh ones, or as far as the prompt pass got when a
            // cancel stopped it part-way (a cancelled request used to be logged and counted as having read them all)
            const int64_t fresh = n - resume;
            const int64_t read_n = cancelled ? std::clamp<int64_t>(pp_reached - resume, 0, fresh) : fresh;
            // DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered> <reused> [hits] [lookups]
            //      [RAM blobs] [file blobs] [file MB]   (CS-T tiers; appended, so an older server reads the rest)
            //      [prompt tokens read]   (#471: fewer than <prompt> - <reused> when a cancel stopped the read)
            //      [offloaded]   (#588: the decode's routed experts the GPU read over PCIe or another GPU computed;
            //                    not in [lookups])
            //      [lookup-chain accepted] [lookup-chain offered] [suffix accepted] [suffix offered] [windows]
            //      (only with --lookup-chain)
            char chain_txt[128] = "";
            if (o.lookup_chain > 0)
                std::snprintf(chain_txt, sizeof(chain_txt), " %lld %lld %lld %lld %lld", (long long) chain_ok,
                              (long long) chain_drafts, (long long) sfx_ok, (long long) sfx_drafts, (long long) dec_windows);
            if (yielded_at >= 0) std::printf("YIELDED %d %lld\n", yielded_slot, (long long) yielded_at);
            std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld %lld %lld %.1f %lld %lld%s\n",
                        (long long) produced_n,
                        (long long) n, prompt_ms, decode_ms, finish, (long long) draft_accepted, (long long) draft_offered,
                        (long long) resume, (long long) req_hits, (long long) req_look,
                        (long long) (src.ram_reads() - ram0), (long long) (src.file_reads() - files0),
                        (double) (src.file_read_bytes() - file_bytes0) / 1e6, (long long) read_n,
                        (long long) req_offload, chain_txt);
            std::fflush(stdout);
            if (admit_slot >= 0) {   // --batch: BADM <slot> <1 = continues in the batch windows | 0 = done>
                bool cont = !cancelled && produced_n == 1 && admit_max_new > 1 && std::strcmp(finish, "length") == 0 &&
                            (int64_t) live.size() == p;
                const auto tc0 = Clock::now();
                if (cont && !copy_to_slot(admit_slot, live, err)) {
                    std::fprintf(stderr, "strata serve: batch admission failed: %s\n", err.c_str());
                    err.clear();
                    bs[(size_t) admit_slot].cached = false;   // its session may be half written
                    cont = false;
                }
                if (cont) {
                    ver.set_slot_sampling(admit_slot, req_sp);   // the request's own sampling, row by row
                    BSlot& sl = bs[(size_t) admit_slot];
                    sl = BSlot{};
                    sl.active = true;
                    sl.x = x;
                    sl.p = p;
                    sl.produced = 1;
                    sl.max_new = admit_max_new;
                    sl.t0 = Clock::now();
                    sl.ids = live;
                    sl.cvec = cvec_cached;
                    sl.img = !live_imgs.empty();   // pictures: not matched again by tokens alone, so not cached
                    const ConvCheckpoint* best = nullptr;
                    for (const ConvCheckpoint& c : checks)
                        if (c.ids.size() < live.size() && (best == nullptr || c.ids.size() > best->ids.size()) &&
                            std::equal(c.ids.begin(), c.ids.end(), live.begin()))
                            best = &c;
                    if (best != nullptr && !sl.img) sl.checks.push_back(*best);
                    std::fprintf(stderr, "strata batch: slot %d takes %lld tokens (copied in %.1f ms)\n", admit_slot,
                                 (long long) live.size(),
                                 std::chrono::duration<double, std::milli>(Clock::now() - tc0).count());
                }
                std::printf("BADM %d %d\n", admit_slot, cont ? 1 : 0);
                std::fflush(stdout);
            }
            if (drive.routing != nullptr) std::fflush(drive.routing);   // the routing trace survives a crash and is watchable mid-session
            // "12288 of 98179" when cancelled mid-read (#471), the rate from what was read
            char read_txt[64];
            if (cancelled)
                std::snprintf(read_txt, sizeof(read_txt), "%lld of %lld", (long long) read_n, (long long) fresh);
            else
                std::snprintf(read_txt, sizeof(read_txt), "%lld", (long long) fresh);
            std::fprintf(stderr, "strata serve: prompt %lld tokens = %lld reused + %s read in %.0f ms (%.1f tok/s), "
                                 "%lld generated in %.0f ms (%.1f tok/s), drafts accepted %lld of %lld, %zu checkpoints%s\n",
                         (long long) n, (long long) resume, read_txt, prompt_ms,
                         prompt_ms > 0 ? 1000.0 * (double) read_n / prompt_ms : 0.0, (long long) produced_n, decode_ms,
                         decode_ms > 0 ? 1000.0 * produced_n / decode_ms : 0.0, (long long) draft_accepted,
                         (long long) draft_offered, checks.size(), cancelled ? " (cancelled)" : "");
            // the VRAM share of the experts the pool looked up while decoding; experts it sent over PCIe for the GPU
            // to read (--pcie-frac) are in neither count - #588: so that share is said beside it (raising --pcie-frac
            // raises the hit rate while the PCIe reads may make the decode slower)
            if (req_look > 0) {
                char off[160] = "";
                if (req_offload > 0)
                    std::snprintf(off, sizeof off, "; %lld more read by the GPU over PCIe or from another GPU (%.1f%% of "
                                  "all %lld routed)", (long long) req_offload,
                                  100.0 * (double) req_offload / (double) (req_look + req_offload),
                                  (long long) (req_look + req_offload));
                std::fprintf(stderr, "strata serve: decode expert cache hit rate: %.1f%% (%lld hits / %lld lookups)%s\n",
                             100.0 * (double) req_hits / (double) req_look,
                             (long long) req_hits, (long long) req_look, off);
            }
            // STRATA_SPLIT_TIMING: where each verify stage's host time went, cumulative per window since the start
            // (waiting for its GPU to ring a layer, the CPU pool and plan per layer, staging the window)
            if (static const bool st_timing = std::getenv("STRATA_SPLIT_TIMING") != nullptr; st_timing)
                for (int st = 0; st < n_stages; ++st) {
                    const strata::core::Verifier& v = stage_ver(st);
                    const double w = v.windows > 0 ? (double) v.windows : 1.0;
                    std::fprintf(stderr, "strata serve: stage %d: %lld windows; per window: wait for the GPU %.3f ms, "
                                         "pool + plan %.3f ms, host staging %.3f ms, commit %.3f ms\n", st,
                                 (long long) v.windows, v.ms_wait / w, v.ms_pool / w, v.ms_host / w, v.ms_commit / w);
                }
            if (g.n_qsa_layers() > 0 && ss.qsa_states[ss.qsa_primary()].kv_mode == 1) {
                // KV streaming, cumulative over the process: blocks the selections named vs blocks read from RAM
                uint64_t miss = 0, look = 0;
                bool over = false;
                for (int64_t i = ss.qsa_ord0; i < ss.qsa_ord0 + ss.qsa_alloc; ++i) {
                    const strata::kernels::KvStreamCounters c = strata::kernels::kv_stream_counters(ss.qsa_states[i].map);
                    miss += c.misses; look += c.lookups; over = over || c.overflow;
                }
                std::fprintf(stderr, "strata serve: KV streaming: %.2f%% of %llu block reads hit VRAM, %.1f MiB read "
                                     "from RAM%s\n", look ? 100.0 * (double) (look - miss) / (double) look : 100.0,
                             (unsigned long long) look, (double) miss * 4224.0 / 1048576.0,
                             over ? " - OVERFLOW (too few resident cells)" : "");
            }
            if (sfx_windows > 0)
                std::fprintf(stderr, "strata serve: suffix drafts: %lld windows, %lld of %lld drafts accepted\n",
                             (long long) sfx_windows, (long long) sfx_ok, (long long) sfx_drafts);
            if (o.lookup_chain > 0)
                std::fprintf(stderr, "strata serve: lookup chain: %lld of %lld windows, %lld of %lld chained drafts accepted\n",
                             (long long) chain_windows, (long long) dec_windows, (long long) chain_ok, (long long) chain_drafts);
            if (strata::core::MtpDrafter::top2_env())
                std::fprintf(stderr, "strata serve: mtp top2: rejected/runner-up by depth %lld/%lld %lld/%lld %lld/%lld %lld/%lld "
                                     "of %lld windows\n", (long long) t2_rej[0], (long long) t2_hit[0], (long long) t2_rej[1],
                             (long long) t2_hit[1], (long long) t2_rej[2], (long long) t2_hit[2], (long long) t2_rej[3],
                             (long long) t2_hit[3], (long long) dec_windows);
            for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0)
                std::fprintf(stderr, "strata serve: CUDA%d: %lld expert entries, %lld active layer launches, %.1f MiB returned "
                                     "(%.1f MiB with full rows) in this request; host %.0f ms staging+launching, %.0f ms "
                                     "waiting for it (since start)\n", r + 1,
                             (long long) (remote_experts[(size_t) r].computed() - remote_before[(size_t) r]),
                             (long long) (remote_experts[(size_t) r].launched_layers() - launches_before[(size_t) r]),
                             (double) (remote_experts[(size_t) r].returned_bytes() - compact_before[(size_t) r]) / 1048576.0,
                             (double) (remote_experts[(size_t) r].full_row_bytes() - full_before[(size_t) r]) / 1048576.0,
                             remote_experts[(size_t) r].ms_begin(), remote_experts[(size_t) r].ms_wait());
        }
        // --conversation-save: the parked conversations, then the live one (the newest file)
        if (disk.enabled()) {
            for (const auto& image : conversations.entries()) save_one(image, "exit");
            if (live_ok && !live.empty()) {
                (void) conversations.take_reuse();   // its RAM goes before the live session's capture
                const strata::core::ConversationView view{live, live_imgs, checks, cvec_cached};
                std::string de;
                auto capture = [&](strata::core::SavedConversation& image) {
                    const uint64_t floor = (uint64_t) o.conversation_cache_min_free_mib * 1024 * 1024;
                    size_t estimate = 0;
                    if (!ver.wait_commit(de) || !snapshot_bytes(view, estimate, de)) return false;
                    if (!strata::core::conversation_memory_admit(strata::core::conversation_available_memory(),
                                                                 estimate, floor)) {
                        de = "physical RAM admission";
                        return false;
                    }
                    return conv_stages.empty()
                               ? strata::core::conversation_snapshot_save(image, view, ss, g, mtp.kv_state(), de)
                               : strata::core::conversation_split_save(image, view, conv_stages, g, mtp.kv_state(), de);
                };
                try {
                    strata::core::SavedConversation image;
                    if (capture(image)) save_one(image, "exit");
                    else std::fprintf(stderr, "strata serve: conversation save: live session not written (%s)\n",
                                      de.c_str());
                } catch (const std::bad_alloc&) {
                    std::fprintf(stderr, "strata serve: conversation save: live session not written (allocation "
                                         "failed)\n");
                }
            }
        }
        save_profile("exit");   // #477: QUIT, or the server closed stdin
        return 0;
    }

    // ---- plan v0.3 P5: the prompt's conditioning positions [0, n_prompt - 1) in batched chunks.  The token loop
    // then starts at the last prompt position, whose prediction is the first generated token.
    int64_t pos_start = 0;
    int64_t spec_pos = -1;  // plan v0.3 P6: where the speculative loop starts (-1 = not used; a native pack's
                            // one-token prompt starts it at 0)
    strata::prefill::Prefill prefill;
    double prefill_batched_ms = 0;
    std::FILE* final_r = o.dump_final_r.empty() ? nullptr : std::fopen(o.dump_final_r.c_str(), "wb");
    std::vector<float> final_r_host(final_r ? (size_t) (g.hc * g.n_embd) : 0);
    std::vector<std::pair<int32_t, int32_t>> lent;     // (residency index, slot) lent to the prompt path
    if (o.prefill_chunk > 0 && n_prompt > 1) {
        void* borrow = nullptr;
        uint64_t borrow_bytes = 0;
        if (!o.no_prefill_borrow && !host_res.empty() && d_res != nullptr) {
            int64_t chunk = o.prefill_chunk;
            int64_t k = plan_lend(chunk);             // auto: the largest chunk that fits; fixed: halved to fit
            if (k > 0 && chunk > (n_prompt - 1 + 255) / 256 * 256) {   // no bigger than the prompt needs
                chunk = std::max<int64_t>(256, (n_prompt - 1 + 255) / 256 * 256);
                k = o.prefill_auto ? auto_slots(chunk) : lend_slots(chunk);
                if (!o.prefill_auto) o.prefill_chunk = chunk;
            }
            if (o.prefill_auto) {
                o.prefill_chunk = k > 0 ? chunk : 1024;
                std::fprintf(stderr, "strata generate: prompt chunk auto: %lld tokens\n", (long long) o.prefill_chunk);
            } else if (chunk != o.prefill_chunk) {
                k = 0;                                 // a fixed chunk that does not fit: its own buffers, as before
            }
            const int64_t blob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
            if (k > 0) {   // the lent slots are refilled after the prompt
                const int32_t first = (int32_t) (xcache.slots() - k);
                for (size_t i = 0; i < host_res.size(); ++i)
                    if (host_res[i] >= first) {
                        lent.emplace_back((int32_t) i, host_res[i]);
                        host_res[i] = strata::core::kNotResident;
                    }
                strata::gpu::copy(d_res, host_res.data(), host_res.size() * sizeof(int32_t));
                borrow = xcache.device_slot(first);
                borrow_bytes = xcache.slot_offsets() ? (uint64_t) (xcache.bytes() - (int64_t) xcache.slot_offsets()[first])
                                                     : (uint64_t) k * (uint64_t) blob;
                std::fprintf(stderr, "strata generate: prompt path borrows %lld cache slots (%.2f GiB)\n", (long long) k,
                             (double) borrow_bytes / 1073741824.0);
            }
        }
        if (borrow == nullptr)
            std::fprintf(stderr, "strata generate: prompt path allocates its own buffers (no cache slots to borrow)\n");
        if (!prefill.init(wt, g, ss, srcp, o.expert_cache > 0 ? &xcache : nullptr,
                          host_res.empty() ? nullptr : host_res.data(), o.prefill_chunk, main_cs, err, borrow,
                          borrow_bytes)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        if (!o.mtp.empty()) {
            if (!mtp.bind(wt, &native_head, nullptr, err)) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            prefill.on_chunk = [&](const float* R_rows, int64_t T, int64_t p0, std::string& e) -> bool {
                // cell i pairs R_i with the token at i + 1 (every such token is in the prompt)
                std::vector<int32_t> nxt((size_t) T);
                for (int64_t t = 0; t < T; ++t) nxt[(size_t) t] = (int32_t) o.tokens[(size_t) (p0 + t + 1)];
                if (prefill.draft_kv(mtp, R_rows, nxt.data(), T, p0, e)) return true;   // E-9
                return e.empty() && mtp.prefill(R_rows, nxt.data(), T, p0, e);
            };
        }
        const Clock::time_point tp0 = Clock::now();
        const int64_t n_batched = (o.prefill_until > 0 && o.prefill_until < n_prompt - 1) ? o.prefill_until : n_prompt - 1;
        if (!prefill.run(o.tokens.data(), n_batched, 0, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        // refill the lent slots from the arena and give them back to the decode tier
        if (!lent.empty()) {
            const Clock::time_point tr = Clock::now();
            for (const auto& [i, slot] : lent) {   // D-4: queued, one wait (STRATA_REFILL_BLOCKING=1: each)
                const bool tb = srcp->transient(i / g.n_expert, i % g.n_expert);   // see the serve loop's refill
                const uint8_t* b = blob_to_copy(srcp, i / g.n_expert, i % g.n_expert, fill_tmp);
                const int64_t nb = (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(i / g.n_expert);
                if (b == nullptr || !(refill_blocking() || tb ? xcache.fill_slot_blocking(slot, b, err, nb)
                                                              : xcache.fill_slot_queued(slot, b, err, nb))) {
                    std::fprintf(stderr, "strata generate: refilling a lent slot failed: %s\n", err.c_str());
                    return 1;
                }
                host_res[(size_t) i] = slot;
            }
            if (!xcache.sync_queued(err)) {
                std::fprintf(stderr, "strata generate: refilling the lent slots failed: %s\n", err.c_str());
                return 1;
            }
            strata::gpu::copy(d_res, host_res.data(), host_res.size() * sizeof(int32_t));
            std::fprintf(stderr, "strata generate: %zu lent slots refilled in %.1f ms\n", lent.size(),
                         std::chrono::duration<double, std::milli>(Clock::now() - tr).count());
        }
        prefill_batched_ms = std::chrono::duration<double, std::milli>(Clock::now() - tp0).count();
        prefill_ms += prefill_batched_ms;
        pos_start = n_batched;
        tok = o.tokens[(size_t) pos_start];
        // the PLE window of the token path: the two tokens before `pos_start`
        ss.ple_prev[0] = pos_start >= 2 ? (int32_t) o.tokens[(size_t) (pos_start - 2)] : -1;
        ss.ple_prev[1] = pos_start >= 1 ? (int32_t) o.tokens[(size_t) (pos_start - 1)] : -1;
        const strata::prefill::PrefillStats& ps = prefill.stats();
        std::fprintf(stderr, "strata generate: prefill %lld tokens in %lld chunks, %.1f ms (%.1f tok/s); experts "
                             "streamed %lld (%lld by DMA, host %.1f ms), resident %lld; PLE %.1f ms\n",
                     (long long) ps.tokens, (long long) ps.chunks, ps.ms_total,
                     ps.ms_total > 0 ? 1000.0 * (double) ps.tokens / ps.ms_total : 0.0, (long long) ps.experts_streamed,
                     (long long) ps.experts_dma, ps.ms_experts_host, (long long) ps.experts_resident, ps.ms_ple);
    }

    for (int64_t pos = pos_start;; ++pos) {
        // plan v0.3 P6: a native pack's last prompt token is the first verify window (T = 1)
        if (native_pack) { spec_pos = pos; break; }
        if (pos >= o.max_context) {
            std::fprintf(stderr, "strata generate: ran out of context at position %lld\n", (long long) pos);
            return 2;
        }
        // **THE TOKEN TIMER STARTS HERE, BEFORE ANY OF THE TOKEN'S WORK (A7).**  It used to start after
        // `put_input`/`embed_row`, which excluded the embedding and the PLE window advance from the reported
        // rate, and it stopped before the NaN scan, the logits dump and the sampler.  The published tok/s
        // figure is a WALL-CLOCK rate: everything one token costs, PLE advance to sampled id.  A rate that
        // excludes real per-token work is not a rate anyone can plan against.
        const Clock::time_point t0 = Clock::now();
        // **THE PLE'S TOKEN WINDOW ADVANCES HERE, ONCE PER TOKEN, AND `ple_stage_token` RUNS OUTSIDE THE
        // CAPTURE.**  Both are the driver's job: `ngram_rows` is a host hash over the last three tokens and the
        // table gather is a host read, so either one inside a captured graph would run once at capture time and
        // replay forever.  `ple_prev` is OLDEST FIRST and `-1` means "no predecessor", which `ngram_rows`
        // treats as the EOS cut - a sequence boundary.
        ss.ple_token = (int32_t) tok;
        Clock::time_point tp = Clock::now();
        // Plan v0.3 P2: the 16 SSD reads start here and complete while the embedding is staged; `ms_ple` is
        // the issue plus the time still spent WAITING afterwards, i.e. the part the embedding did not hide.
        if (ss.ple.ready() && !strata::core::ple_issue_token(ss.ple, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        {
            const Clock::time_point n = Clock::now();
            ms_ple += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }
        if (!put_input(tok, pos)) return 1;
        {
            const Clock::time_point n = Clock::now();
            ms_embed += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }
        if (ss.ple.ready() && !strata::core::ple_finish_token(ss.ple, token_stream, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        {
            const Clock::time_point n = Clock::now();
            ms_ple += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }

        if (pos % 256 == 0 || pos + 1 >= n_prompt - 1)
            std::fprintf(stderr, "strata generate: position %lld, token %lld%s\n", (long long) pos, (long long) tok,
                         pos < n_prompt ? " (prompt)" : "");
        // **`d.layers` IS THE BLOB'S LAYER AXIS, NOT A COUNTER.**  The adapter uses it to index
        // `experts.bin` as `layer * n_expert + expert`, so it MUST restart at 0 for every token.  Leaving it
        // running across tokens asks for layer 48 of a 48-layer file on the second token - which
        // `FileExpertSource` REFUSES rather than wrapping into layer 0's experts, and that refusal is the only
        // reason this was a clean error instead of a silently wrong second token.
        drive.d.layers = 0;
        drive.d.experts = 0;
        drive.d.failed = false;
        err.clear();
        if (o.no_capture) {
            if (!strata::core::session_token(wt, g, pos, /*pos_base=*/0, ss, d_parts, main_cs,
                                             o.sync_every_layer, err)) {
                std::fprintf(stderr, "strata generate: session_token: %s\n", err.c_str());
                return 1;
            }
        } else {
            strata::core::doorbell_reset(db);
            if (tgraph.captured) {
                if (!strata::core::session_run_token(g, pos, /*pos_base=*/0, ss, tgraph, pool_fn, pool_user,
                                                     loop_scratch.y_miss, main_cs, err)) {
                    std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                    return 1;
                }
            } else if (!strata::core::session_loop(g, pos, /*pos_base=*/0, ss, gr, pool_fn, hit_fn, pool_user, /*overlap=*/true, main_cs,
                                        err, layer_stage, &loop_scratch)) {
                std::fprintf(stderr, "strata generate: session_loop: %s\n", err.c_str());
                return 1;
            }
        }
        if (final_r != nullptr) {
            strata::gpu::copy(final_r_host.data(), ss.R, final_r_host.size() * sizeof(float));
            const int64_t posrec[2] = {pos, tok};
            std::fwrite(posrec, sizeof posrec, 1, final_r);
            std::fwrite(final_r_host.data(), sizeof(float), final_r_host.size(), final_r);
        }
        if (drive.d.failed) {
            std::fprintf(stderr, "strata generate: the expert pool failed at layer %lld expert %lld: %s\n",
                         (long long) drive.d.fail_layer, (long long) drive.d.fail_expert,
                         drive.d.fail ? drive.d.fail : "(no message)");
            return 1;
        }
        {
            // **THE LAYER LOOP ITSELF, WHICH IS WHAT `--gpu-only-full` HAS TO BE COMPARED AGAINST.**
            const Clock::time_point n = Clock::now();
            ms_layers += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }
        // ---- the ladder for THIS position, in the order `session_loop` filled it: layer 0 first.
        if (layer_dump != nullptr) std::fwrite(layer_stage, sizeof(float), layer_floats, layer_dump);
        if (half_dump != nullptr) {
            std::fwrite(half_stage, sizeof(float), (size_t) g.n_layers * (size_t) half_stride, half_dump);
        }
        if (!run_head(token_stream)) {
            std::fprintf(stderr, "strata generate: lm_head: %s\n", err.c_str());
            return 1;
        }
        // **A CHECKPOINT AFTER THE HEAD, BECAUSE AN ASYNC FAULT IS STICKY AND LIES ABOUT WHERE IT HAPPENED.**
        // Measured, and it cost an hour: without this, `embed_row`'s D2H on the NEXT token reported "an illegal
        // memory access" at a plane offset that has nothing to do with the fault, and the layer that actually
        // faulted had completed its own error checks successfully - because its kernels had not run yet.  A
        // sticky error surfaces at the next SYNCHRONISING call, which is whatever happens to come next.
        if (!o.stream_token && !strata::gpu::device_sync()) {
            std::fprintf(stderr, "strata generate: the device faulted in lm_head at position %lld: %s\n",
                         (long long) pos, strata::gpu::last_error());
            return 1;
        }
        {
            const Clock::time_point n = Clock::now();
            ms_head += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }
        const bool emit_logits = dump != nullptr &&
            strata::program::logits_selection::selected(pos - dump_first, dump_positions, o.logits_stride);
        const bool read_logits = !o.stream_token || o.check_logits || emit_logits;
        if (read_logits && (!strata::gpu::copy_async(logits.data(), d_logits, (size_t) n_vocab * 4, token_stream) ||
                            !strata::gpu::stream_sync(token_stream))) {
            std::fprintf(stderr, "strata generate: reading the logits back failed\n");
            return 1;
        }
        int bad = 0;
        if (read_logits) for (float v : logits) if (!std::isfinite(v)) ++bad;
        if (bad != 0) {
            std::fprintf(stderr, "strata generate: %d of %lld logits are not finite at position %lld\n", bad,
                         (long long) n_vocab, (long long) pos);
            return 1;
        }
        if (emit_logits && std::fwrite(logits.data(), sizeof(float), (size_t) n_vocab, dump) != (size_t) n_vocab) {
            std::fprintf(stderr, "strata generate: cannot write logits at position %lld\n", (long long) pos);
            std::fclose(dump);
            return 1;
        }
        {
            // **993 KB OF SYNCHRONOUS D2H AND A 248,320-FLOAT HOST SCAN, EVERY TOKEN.**  (The review's notes
            // say 151,936 floats; the artifact's `output.weight` is 248,320 rows, so the real figure is 1.6x
            // that - a number nobody had checked because nothing measured this term.)  R2.6 asks for the dump
            // and the scan to be behind flags; round 36 did exactly that and measured it SLOWER, because on
            // this driver a large blocking readback is also what flushes the pipeline for the sampler that
            // follows.  Timed so the claim can be re-checked rather than remembered.
            const Clock::time_point n = Clock::now();
            ms_readback += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }

        int next = 0;
        // The draw is Philox(seed, position), as in a verify window (row t at pos0 draws pos0 + t): a seed gives
        // the same text whether a token comes from this path or from the speculative loop below.
        sp.counter = (uint64_t) pos;
        strata::kernels::sample_tokens(d_logits, 1, (int) n_vocab, nullptr, 0, sp, d_next, token_stream);
        if (!strata::gpu::copy_async(&next, d_next, sizeof(int), token_stream) ||
            !strata::gpu::stream_sync(token_stream)) {
            std::fprintf(stderr, "strata generate: reading the sampled token back failed: %s\n",
                         strata::gpu::last_error());
            return 1;
        }
        // The sampled-token synchronization also completes every captured QSA
        // status readback. Retain one status per layer so a later layer cannot
        // hide an earlier failure; no extra synchronization or token allocation.
        if (o.native_flash_attn_short) for (int64_t i = ss.qsa_ord0; i < ss.qsa_ord0 + ss.qsa_alloc; ++i) {
            const int32_t status = ss.qsa_states[i].host_step[strata::kernels::kStepCount];
            if (status != 0) {
                std::fprintf(stderr, "strata generate: native attention status %d at QSA layer %lld, position %lld\n",
                             status, (long long) i, (long long) pos);
                return 1;
            }
        }
        if (next < 0 || next >= n_vocab) {
            std::fprintf(stderr, "strata generate: the sampler returned %d, outside 0..%lld\n", next,
                         (long long) (n_vocab - 1));
            return 1;
        }
        {
            // **TWO DEVICE-WIDE SYNCS FOR FOUR BYTES.**  `sample_tokens(nullptr)` ends in
            // `strata::gpu::device_sync()` (`sampler.cu:245`) and the blocking 4-byte read below is the second.
            const Clock::time_point n = Clock::now();
            ms_sample += std::chrono::duration<double, std::milli>(n - tp).count();
            ++phase_tokens;
        }
        // CHARGED HERE, AFTER THE SAMPLER, so the wall-clock rate covers the whole token including the embedding,
        // the NaN scan, the logits readback and the sample (A7).  Only DECODE positions count; prefill is
        // measured separately.
        if (pos >= n_prompt - 1) total_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        else prefill_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        if (pos == n_prompt - 1) ttft_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
        // **THE PREDICTION AT THE LAST PROMPT POSITION *IS* THE FIRST GENERATED TOKEN.**  Sampling on every
        // position and recording only from `n_prompt - 1` onward is what keeps the two cases from needing
        // separate handling - and the version that "obviously" only samples after the prompt loses exactly one
        // token's worth of conditioning.
        if (pos >= n_prompt - 1) produced.push_back(next);
        if ((int64_t) produced.size() >= o.max_new) break;
        if (o.stop_eos && pos >= n_prompt - 1 &&
            std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) next) != o.eos_ids.end()) break;
        // TEACHER FORCING while the prompt lasts: the next input is the prompt's own next token, not the
        // model's guess.  Feeding the guess would make the run depend on the model's own errors from position
        // 1, which is a different (and worse) measurement of the same prompt.
        // and the window advances: the token just decoded becomes the newest predecessor.
        ss.ple_prev[0] = ss.ple_prev[1];
        ss.ple_prev[1] = (int32_t) tok;
        tok = (pos + 1 < n_prompt) ? o.tokens[(size_t) (pos + 1)] : next;
        // Plan v0.3 P6: from the first generated token on, the speculative loop below takes over.
        if (o.spec > 0 && pos >= n_prompt - 1) { spec_pos = pos + 1; break; }
    }

    // ================================ plan v0.3 P6: SPECULATIVE DECODING ================================
    //
    // Each round verifies [the last emitted token, drafts...] in one window; the window's argmax after token t
    // is exactly what greedy decode would emit there, so the first draft that differs ends the round and the
    // round emits (accepted drafts + 1) tokens.  `commit` keeps the state of the tokens that were emitted.
    const bool ended = o.stop_eos && !produced.empty() &&
                       std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) produced.back()) != o.eos_ids.end();
    if (spec_pos >= 0 && (int64_t) produced.size() < o.max_new && !ended) {
        auto read_ids = [](const std::string& path, std::vector<int64_t>& ids) {
            std::ifstream in(path);
            std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            std::string e;
            return (bool) in && parse_i64_list(text.c_str(), ids, e);
        };
        std::vector<int64_t> oracle, follow;
        if (!o.spec_oracle.empty() && !read_ids(o.spec_oracle, oracle)) {
            std::fprintf(stderr, "strata generate: cannot read --spec-oracle %s\n", o.spec_oracle.c_str());
            return 2;
        }
        if (!o.spec_follow.empty() && !read_ids(o.spec_follow, follow)) {
            std::fprintf(stderr, "strata generate: cannot read --spec-follow %s\n", o.spec_follow.c_str());
            return 2;
        }
        // --spec-follow: the run ends with the continuation
        const int64_t max_new = follow.empty() ? o.max_new : std::min<int64_t>(o.max_new, (int64_t) follow.size());
        int64_t follow_differ = 0, follow_emitted = 0;
        if (thits.d_res == nullptr) {
            std::fprintf(stderr, "strata generate: --spec needs the device residency table (--expert-profile, "
                                 "--expert-cache and the token graph)\n");
            return 2;
        }
        mem_mark("the head and the prompt path");
        strata::core::Verifier ver;
        strata::core::VerifyHits vh;
        vh.d_res = thits.d_res;
        vh.cache_base = thits.cache_base;
        vh.blob = thits.blob;
        vh.slot_off = xcache.slot_offsets();   // E-6: the device plan's pointers
        vh.n_slots = xcache.slots();
        vh.h_res = host_res.empty() ? nullptr : host_res.data();   // every expert resident: windows without the host
        if (!ver.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, o.spec, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        const bool use_mtp = !o.mtp.empty();
        if (use_mtp && !mtp.bind(wt, &native_head, ver.final_R_all(), err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        mem_mark("the verifier and the drafter's binding");
        if (use_mtp) mtp.set_draft_sampling(sp);
        ver.set_sampling(sp);   // the CLI's own sampling (until 0.1.19 this loop was always greedy); no penalties here
        ver.set_split(o.spec_split);
        // auto: the copy kernel for every pack.  DMA (the native packs' default until 0.1.13) has the host call
        // cudaMemcpyAsync + cudaLaunchHostFunc inside a verify window while the GPU spins on the flag they raise;
        // issue #31's thread dumps show the host stuck in that cudaMemcpyAsync on a driver lock for good.  The copy
        // kernel needs no host CUDA call there, and costs ~1-3% decode on IQ3_S (45.3 -> 44.8 tok/s, 8 requests).
        ver.set_pcie_mode(o.pcie_mode == "dma" ? 0 : o.pcie_mode == "direct" ? 1 : 2);
        drive.d.plan = ver.plan_sink();
        drive.d.pcie_num = (int) (o.pcie_frac * 256.0 + 0.5);
        if (drive.d.pcie_num < 0) drive.d.pcie_num = 0;
        if (drive.d.pcie_num > 256) drive.d.pcie_num = 256;
        const int64_t pcie0 = drive.d.pcie_experts;
        if (o.adapt_every > 0 && o.adapt_swaps > 0) drive.d.usage.assign((size_t) (g.n_layers * g.n_expert), 0.0f);
        int64_t swaps_total = 0;
        double ms_adapt = 0;
        strata::gpu::Stream adapt_stream = nullptr;
        if (!drive.d.usage.empty() && !(adapt_stream = strata::gpu::stream_create())) {
            std::fprintf(stderr, "strata generate: cannot create the refill stream\n");
            return 1;
        }
        // plan v0.3 P6: swaps in flight - (residency index, slot) admitted when adapt_ev has completed
        std::vector<std::pair<int32_t, int32_t>> pending;
        strata::gpu::Event* adapt_ev = nullptr;
        strata::gpu::event_create(&adapt_ev);
        int64_t adapt_rounds = 0;   // counted here: `rounds` is declared below the adapt lambda
        // STRATA_TRACE_ADAPT=1: whether a round's copies had landed when the next window read the table, and why an
        // adapt round did or did not swap
        static const bool trace_adapt = std::getenv("STRATA_TRACE_ADAPT") != nullptr;
        auto apply_pending = [&](bool wait) {
            if (pending.empty()) return;
            if (wait) strata::gpu::event_sync(adapt_ev);
            else if (!strata::gpu::event_query(adapt_ev)) {
                if (trace_adapt)
                    std::fprintf(stderr, "strata: PENDING not landed, %zu stay non-resident this window\n",
                                 pending.size());
                return;
            }
            if (trace_adapt)
                std::fprintf(stderr, "strata: PENDING landed, %zu experts become resident\n", pending.size());
            for (const auto& [i, slot] : pending) {
                host_res[(size_t) i] = slot;
                srcp->release((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);   // in VRAM now: RAM not needed
            }
            src.commit_exchanges();
            pending.clear();
            if (d_res != nullptr)
                strata::gpu::copy(d_res, host_res.data(), host_res.size() * sizeof(int32_t));
        };
        // Plan v0.3 P6: the VRAM tier follows the conversation.  Candidates are missing experts routed at least
        // twice (decayed); each is paired with its layer's least-routed resident expert and swapped when it was
        // routed clearly more often.  Copies run between rounds, when the GPU is idle.
        auto adapt = [&]() -> bool {
            const Clock::time_point ta = Clock::now();
            ++adapt_rounds;
            if (!pending.empty()) {
                if (trace_adapt)
                    std::fprintf(stderr, "strata: ADAPT round=%lld SKIPPED, %zu swaps still in flight\n",
                                 (long long) adapt_rounds, pending.size());
                return true;   // the previous swaps are still in flight
            }
            if (trace_adapt) std::fprintf(stderr, "strata: ADAPT round=%lld considering\n", (long long) adapt_rounds);
            struct Swap { float gain; int32_t layer, in, out; };
            std::vector<Swap> swaps;
            std::vector<std::pair<float, int32_t>> cand, vict;
            for (int64_t l = 0; l < g.n_layers; ++l) {
                cand.clear();
                vict.clear();
                const float* u = drive.d.usage.data() + l * g.n_expert;
                const int32_t* r = host_res.data() + l * g.n_expert;
                for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                    if (r[e] < 0) { if (u[e] >= 2.0f && !helper_holds(drive.d, l, e)) cand.emplace_back(u[e], e); }
                    else vict.emplace_back(u[e], e);
                }
                if (cand.empty() || vict.empty()) continue;
                std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
                const size_t nc = std::min(cand.size(), vict.size());
                std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                                  [](auto& a, auto& b) { return a.first < b.first; });
                for (size_t i = 0; i < nc; ++i) {
                    if (cand[i].first < vict[i].first + 1.5f) break;
                    swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
                }
            }
            std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
            if ((int) swaps.size() > o.adapt_swaps) swaps.resize((size_t) o.adapt_swaps);
            if (!resident_stage_swaps(src, host_res, g.n_expert, swaps,   // this loop refills the first card only
                                      [&](int32_t) { return SwapHome{&xcache, adapt_stream, -1}; }))
                return false;
            for (const Swap& s : swaps) {
                const size_t in = (size_t) s.layer * g.n_expert + s.in, out = (size_t) s.layer * g.n_expert + s.out;
                const int32_t slot = host_res[out];
                // asynchronous: the copies run while the MTP drafts; the next window waits for them (a transient
                // source's blob - the GGUF read in place - is copied into fill_tmp, so that swap is synchronous)
                const bool tb = srcp->transient(s.layer, s.in);
                const uint8_t* b = blob_to_copy(srcp, s.layer, s.in, fill_tmp);
                const size_t sn = (size_t) strata::kernels::cpu::expert_layout().blob_bytes(s.layer);
                if (slot < 0 || b == nullptr ||
                    !(tb ? strata::gpu::copy(xcache.device_slot(slot), b, sn)
                         : strata::gpu::copy_async(xcache.device_slot(slot), b, sn, adapt_stream))) {
                    std::fprintf(stderr, "strata generate: an adaptive refill failed\n");
                    return false;
                }
                host_res[out] = strata::core::kNotResident;   // evicted now: the CPU computes it meanwhile
                srcp->prefetch(s.layer, s.out);   // a file-backed arena released its pages: read them back ahead
                pending.emplace_back((int32_t) in, slot);      // resident once the copy has landed
            }
            if (!swaps.empty()) strata::gpu::event_record(adapt_ev, adapt_stream);
            if (trace_adapt)
                std::fprintf(stderr, "strata: ADAPT round=%lld swapped %zu of %d slots, usage decayed\n",
                             (long long) adapt_rounds, swaps.size(), o.adapt_swaps);
            for (float& v : drive.d.usage) v *= o.adapt_decay;
            swaps_total += (int64_t) swaps.size();
            ms_adapt += std::chrono::duration<double, std::milli>(Clock::now() - ta).count();
            return true;
        };
        int64_t p = spec_pos;
        int32_t x = (int32_t) tok;
        std::vector<int32_t> drafts((size_t) o.spec, 0);
        std::vector<float> dprob((size_t) o.spec, 1.0f);
        std::vector<int64_t> window_hist((size_t) o.spec + 1, 0);
        // plan v0.3 P6: with a native pack the first window is the last prompt token alone (it produces the first
        // generated token and the MTP's first cell); otherwise the token loop already did that.
        bool first_window = native_pack;
        if (use_mtp && !first_window &&
            !mtp.draft_first(o.spec, ss.R, x, p - 1, drafts.data(), err, dprob.data(), (float) o.spec_min_p)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::vector<int32_t> window((size_t) o.spec), outv((size_t) o.spec);
        std::vector<float> window_logits;       // --dump-logits: the emitted rows of a window
        std::vector<uint8_t> fdiff((size_t) o.spec, 0);
        std::FILE* hashes = o.window_hashes.empty() ? nullptr : std::fopen(o.window_hashes.c_str(), "w");
        if (!o.window_hashes.empty() && hashes == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.window_hashes.c_str());
            return 1;
        }
        std::vector<float> hash_rows;
        std::vector<int64_t> accepted_hist((size_t) o.spec, 0);
        int64_t rounds = 0, drafts_total = 0, drafts_ok = 0, corrupt_counter = 0;
        const int S_mtp = o.mtp_max_t > 0 ? std::min(o.mtp_max_t, o.spec) : o.spec;
        if (use_mtp && S_mtp < o.spec) mtp.set_max_drafts(S_mtp - 1);
        strata::spec::SuffixDrafter sfx(std::max(1, o.suffix_draft), 64, (size_t) o.max_context + 4096);
        strata::spec::DraftPolicy policy(o.spec);   // MTP or lookup window (see draft_policy.hpp)
        std::vector<int32_t> sbuf((size_t) o.spec, 0);
        int64_t sfx_windows = 0, sfx_drafts = 0, sfx_ok = 0;
        int64_t chain_windows = 0, chain_drafts = 0, chain_ok = 0;   // --lookup-chain's own counts
        std::vector<int32_t> cbuf((size_t) o.spec, 0), ctail;
        strata::spec::PromptLookupSource lookup_src(sfx);
        const bool sfx_on = o.suffix_draft > 0 || o.lookup_chain > 0;
        if (sfx_on) {
            for (int64_t t : o.tokens) sfx.append((int32_t) t);
            for (int64_t t : produced) sfx.append((int32_t) t);
        }
        if (o.lookup_chain > 0) {
            extra_sources_reset();
            for (int64_t t : o.tokens) { const int32_t t32 = (int32_t) t; extra_sources_append(&t32, 1); }
            for (int64_t t : produced) { const int32_t t32 = (int32_t) t; extra_sources_append(&t32, 1); }
        }
        const double pool_ms0 = drive.cpu_ms;
        // pages read back from disk or swap while decoding: the expert arena is not locked, and two runs whose
        // CPU pool was 6x slower came while tens of GB were being downloaded (bench/results/2026-09-30-xe-swift-iq2xs)
        const auto major_faults = [] { rusage u{}; getrusage(RUSAGE_SELF, &u); return (long long) u.ru_majflt; };
        const long long majflt0 = major_faults();
        const int64_t misses0 = drive.d.multi_misses, entries0 = drive.d.multi_entries;
        // the low-RAM mode's tiers during the decode only (the startup fill and the prompt read files too)
        const int64_t file_reads0 = src.file_reads(), ram_reads0 = src.ram_reads();
        const uint64_t file_bytes0 = src.file_read_bytes();
        while ((int64_t) produced.size() < max_new) {
            const Clock::time_point t0 = Clock::now();
            int T = S_mtp;
            if (use_mtp && o.spec_min_p > 0.0) {
                T = 1;
                while (T < S_mtp && dprob[(size_t) T - 1] >= (float) o.spec_min_p) ++T;
            }
            if (first_window) T = 1;
            bool from_sfx = false;
            int sfx_match = 0;
            if (o.suffix_draft > 0 && !first_window) {
                const int k = sfx.propose(o.spec - 1, sbuf.data());
                sfx_match = sfx.last_match();
                if (k > 0 && (!use_mtp || sbuf[0] == drafts[0])) {
                    const strata::spec::DraftPolicy::Pick pk = policy.choose(T, k, sfx_match);
                    if (pk.lookup) { T = pk.t; from_sfx = true; }
                }
            }
            // --lookup-chain: what followed an earlier occurrence of the context + the MTP's drafts, after them
            int chain_n = 0, cm = 0;
            if (o.lookup_chain > 0 && use_mtp && !first_window && !from_sfx && T < o.spec) {
                int csrc = -1;
                const int nt = chain_tail(sfx, drafts.data(), T - 1, ctail);
                chain_n = strata::spec::propose_from_sources(lookup_src, ctail.data(), nt, T - 1,
                                                             std::min(o.lookup_chain, o.spec - T), o.lookup_chain_min,
                                                             cbuf.data(), &cm, &csrc);
                if (chain_n > 0 && std::getenv("STRATA_LOOKUP_CHAIN_FIXED") == nullptr) {
                    double p_mtp = 1.0;
                    for (int i = 0; i < T - 1; ++i) p_mtp *= dprob[(size_t) i];
                    chain_n = policy.chain(T, p_mtp, chain_n, cm);
                }
            }
            const int T_mtp = T;
            T += chain_n;
            const bool timed_round = !first_window;
            ++window_hist[(size_t) T];
            if (p + T > o.max_context) {
                std::fprintf(stderr, "strata generate: ran out of context at position %lld\n", (long long) p);
                return 2;
            }
            window[0] = x;
            for (int i = 1; i < T; ++i) {
                const size_t at = produced.size() - 1 + (size_t) i;
                int32_t d = from_sfx ? sbuf[(size_t) i - 1] : use_mtp ? drafts[(size_t) i - 1]
                                                    : at < oracle.size() ? (int32_t) oracle[at] : 0;
                if (i >= T_mtp) d = cbuf[(size_t) (i - T_mtp)];
                if (o.spec_corrupt > 0 && (++corrupt_counter % o.spec_corrupt) == 0) d = (d + 1) % (int32_t) n_vocab;
                window[(size_t) i] = d;
            }
            drive.d.layers = 0;
            drive.d.experts = 0;
            drive.d.failed = false;
            // #463: the previous adapt round's copies land first - with a non-blocking query, whether a swapped-in
            // expert ran on the GPU or the CPU (they round differently) depended on the copy's timing
            // (STRATA_ADAPT_NOWAIT=1: 0.1.37's non-blocking query, the A/B)
            apply_pending(!adapt_nowait());
            if (!ver.run(T, window.data(), p, &drive_pool_multi, &drive, outv.data(), err)) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            if (drive.d.failed) {
                std::fprintf(stderr, "strata generate: the expert pool failed at layer %lld expert %lld: %s\n",
                             (long long) drive.d.fail_layer, (long long) drive.d.fail_expert,
                             drive.d.fail ? drive.d.fail : "(no message)");
                return 1;
            }
            if (hashes != nullptr) {   // FNV-1a over the window's final residual rows (a test path: a plain copy)
                const size_t n_floats = (size_t) T * (size_t) (g.hc * g.n_embd);
                hash_rows.resize(n_floats);
                if (!strata::gpu::copy(hash_rows.data(), ver.final_R_all(), n_floats * sizeof(float))) {
                    std::fprintf(stderr, "strata generate: --window-hashes: %s\n", strata::gpu::last_error());
                    return 1;
                }
                uint64_t h = 1469598103934665603ull;
                const auto* bytes = (const uint8_t*) hash_rows.data();
                for (size_t i = 0; i < n_floats * sizeof(float); ++i) h = (h ^ bytes[i]) * 1099511628211ull;
                std::fprintf(hashes, "%lld %lld %d %016llx", (long long) rounds, (long long) p, T, (unsigned long long) h);
                for (int i = 0; i < T; ++i) std::fprintf(hashes, " %d", (int) outv[(size_t) i]);
                std::fprintf(hashes, "\n");
            }
            if (!follow.empty()) {   // the continuation stands in for the argmax
                const size_t k0 = produced.size();
                for (int i = 0; i < T && k0 + (size_t) i < follow.size(); ++i) {
                    fdiff[(size_t) i] = outv[(size_t) i] != (int32_t) follow[k0 + (size_t) i];
                    outv[(size_t) i] = (int32_t) follow[k0 + (size_t) i];
                }
            }
            int a = 0;
            while (a < T - 1 && window[(size_t) a + 1] == outv[(size_t) a]) ++a;
            if (first_window) {
                first_window = false;
                ttft_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
            }
            // plan v0.3 P6: the adaptive tier's host work (ranking, copy submission) runs on its own thread while the
            // GPU commits and drafts; it touches only the residency tables, which nothing reads until the next window
            std::thread adapt_thr;
            bool adapt_ok = true;
            if (!drive.d.usage.empty() && ((rounds + 1) % o.adapt_every) == 0)
                adapt_thr = std::thread([&] { adapt_ok = adapt(); });
            if (!ver.commit(a + 1, err)) {
                if (adapt_thr.joinable()) adapt_thr.join();
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            ++rounds;
            drafts_total += T - 1;
            drafts_ok += a;
            ++accepted_hist[(size_t) a];
            if (from_sfx) { ++sfx_windows; sfx_drafts += T - 1; sfx_ok += a; }
            if (chain_n > 0) { ++chain_windows; chain_drafts += chain_n; chain_ok += std::max(0, a - (T_mtp - 1)); }
            bool eos = false;
            int emitted = 0;
            for (int i = 0; i <= a && (int64_t) produced.size() < max_new && !eos; ++i, ++emitted) {
                produced.push_back(outv[(size_t) i]);
                if (sfx_on) sfx.append(outv[(size_t) i]);
                if (o.lookup_chain > 0) extra_sources_append(&outv[(size_t) i], 1);
                eos = o.stop_eos && std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outv[(size_t) i]) != o.eos_ids.end();
                if (!follow.empty()) { follow_differ += fdiff[(size_t) i]; ++follow_emitted; }
            }
            if (dump != nullptr && emitted > 0) {   // the rows of the emitted tokens; the rejected drafts' are not
                window_logits.resize((size_t) emitted * (size_t) n_vocab);
                if (!ver.read_logits(emitted, window_logits.data(), err)) {
                    if (adapt_thr.joinable()) adapt_thr.join();
                    std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                    return 1;
                }
                for (int i = 0; i < emitted; ++i) {
                    if (!strata::program::logits_selection::selected(p + i - dump_first, dump_positions, o.logits_stride))
                        continue;
                    const float* row = window_logits.data() + (size_t) i * (size_t) n_vocab;
                    if (std::fwrite(row, sizeof(float), (size_t) n_vocab, dump) != (size_t) n_vocab) {
                        if (adapt_thr.joinable()) adapt_thr.join();
                        std::fprintf(stderr, "strata generate: cannot write logits at position %lld\n", (long long) (p + i));
                        return 1;
                    }
                }
            }
            if (eos) {
                if (adapt_thr.joinable()) adapt_thr.join();
                total_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                break;
            }
            const bool drafted = !use_mtp || (int64_t) produced.size() >= max_new ||
                                 mtp.draft(T, outv.data(), p, a, drafts.data(), err, dprob.data(), (float) o.spec_min_p);
            if (adapt_thr.joinable()) adapt_thr.join();
            if (!adapt_ok) return 1;
            if (!drafted) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            x = outv[(size_t) a];
            p += a + 1;
            const double round_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            total_ms += round_ms;
            if (timed_round && chain_n == 0) policy.observe(from_sfx, T, a, sfx_match, round_ms);
            else if (timed_round) policy.observe_chain(T_mtp, chain_n, a, cm, round_ms);
            if (rounds % 64 == 0)
                std::fprintf(stderr, "strata generate: position %lld, %lld tokens, %lld rounds\n", (long long) p,
                             (long long) produced.size(), (long long) rounds);
        }
        // the last commit (set_commit_async) before anything reads the session again
        if (!ver.wait_commit(err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        if (hashes != nullptr) std::fclose(hashes);
        std::printf("%-24s %lld rounds of %d, drafts accepted %lld of %lld (%.3f), %.2f tokens per round\n",
                    "speculation", (long long) rounds, o.spec, (long long) drafts_ok, (long long) drafts_total,
                    drafts_total > 0 ? (double) drafts_ok / (double) drafts_total : 0.0,
                    rounds > 0 ? (double) (drafts_ok + rounds) / (double) rounds : 0.0);
        if (!follow.empty())
            std::printf("%-24s %lld of %lld emitted tokens differ from the argmax\n", "follow",
                        (long long) follow_differ, (long long) follow_emitted);
        if (o.spec_min_p > 0.0) {
            std::printf("%-24s", "window sizes");
            for (size_t i = 1; i < window_hist.size(); ++i) std::printf(" T%zu:%lld", i, (long long) window_hist[i]);
            std::printf("  (min draft probability %.2f)\n", o.spec_min_p);
        }
        if (o.suffix_draft > 0)
            std::printf("%-24s %lld windows, drafts accepted %lld of %lld\n", "suffix drafts", (long long) sfx_windows,
                        (long long) sfx_ok, (long long) sfx_drafts);
        if (o.lookup_chain > 0)
            std::printf("%-24s %lld windows, chained drafts accepted %lld of %lld\n", "lookup chain",
                        (long long) chain_windows, (long long) chain_ok, (long long) chain_drafts);
        std::printf("%-24s", "accepted per round");
        for (size_t i = 0; i < accepted_hist.size(); ++i) std::printf(" %zu:%lld", i, (long long) accepted_hist[i]);
        std::printf("\n");
        if (rounds > 0)
            std::printf("%-24s wait for rings %.3f  pool %.3f  host %.3f  commit %.3f ms/round; CPU experts %.2f "
                        "distinct / %.2f routed per layer\n",
                        "verify window", ver.ms_wait / rounds, ver.ms_pool / rounds, ver.ms_host / rounds,
                        ver.ms_commit / rounds,
                        (double) (drive.d.multi_misses - misses0) / (double) (rounds * g.n_layers),
                        (double) (drive.d.multi_entries - entries0) / (double) (rounds * g.n_layers));
        if (rounds > 0 && src.resident_count() > 0)
            std::printf("%-24s %lld blobs from the resident copy in RAM while decoding (%lld experts, %.2f GiB)\n",
                        "resident tier", (long long) (src.ram_reads() - ram_reads0), (long long) src.resident_count(),
                        (double) src.resident_bytes() / 1073741824.0);
        if (rounds > 0 && src.gguf_mode())   // the low-RAM mode's file tier (the GGUF in place)
            std::printf("%-24s %lld blobs assembled from the GGUF files while decoding (%.0f MB, %.1f MB/round)\n",
                        "file tier", (long long) (src.file_reads() - file_reads0),
                        (double) (src.file_read_bytes() - file_bytes0) / 1e6,
                        (double) (src.file_read_bytes() - file_bytes0) / 1e6 / (double) rounds);
        if (rounds > 0) {
            long long swap_kb = -1;
            std::ifstream st("/proc/self/status");
            for (std::string l; std::getline(st, l);)
                if (l.rfind("VmSwap:", 0) == 0) swap_kb = std::atoll(l.c_str() + 7);
            std::printf("%-24s %lld major page faults while decoding; %lld MiB of this process in swap\n",
                        "host memory", major_faults() - majflt0, swap_kb < 0 ? -1 : swap_kb / 1024);
        }
        if (rounds > 0)
            std::printf("%-24s gate/up %.3f  quantize %.3f  down %.3f ms/round; %.1f GB/s over the rows phases; "
                        "CPU pool call %.3f ms/round\n", "pool multi", pool.ms_multi_gu / rounds,
                        pool.ms_multi_q / rounds, pool.ms_multi_down / rounds,
                        (double) pool.multi_bytes / 1e6 / std::max(1e-9, pool.ms_multi_gu + pool.ms_multi_down),
                        (drive.cpu_ms - pool_ms0) / rounds);
        if (rounds > 0)
            std::printf("%-24s plan %.3f  activation quantize %.3f  jobs %.3f  run %.3f ms/round\n", "dispatch",
                        drive.d.ms_plan / rounds, drive.d.ms_actq / rounds, drive.d.ms_jobs / rounds,
                        drive.d.ms_run / rounds);
        if (rounds > 0 && !drive.d.usage.empty())
            std::printf("%-24s %lld experts swapped into the VRAM tier (every %d rounds, %.3f ms/round)\n", "adaptive tier",
                        (long long) swaps_total, o.adapt_every, ms_adapt / rounds);
        if (rounds > 0 && drive.d.pcie_num > 0)
            std::printf("%-24s %.2f distinct experts per layer read over PCIe (share %d/256 of the misses)\n",
                        "pcie experts", (double) (drive.d.pcie_experts - pcie0) / (double) (rounds * g.n_layers),
                        drive.d.pcie_num);
        (void) pool_ms0;
        if (use_mtp && rounds > 0)
            std::printf("%-24s %.3f ms/round drafting (%lld rounds), MTP prompt %.1f ms, %.0f MiB of VRAM\n", "mtp",
                        mtp.ms_draft / (double) mtp.rounds, (long long) mtp.rounds, mtp.ms_prefill,
                        (double) mtp.vram_bytes() / 1048576.0);
    }

    if (dump != nullptr && std::fclose(dump) != 0) {
        std::fprintf(stderr, "strata generate: cannot finish logits dump\n");
        return 1;
    }
    if (layer_dump != nullptr) {
        std::fclose(layer_dump);
        strata::gpu::free(layer_stage);
        std::printf("%-24s %s (%lld layers + the input x %d streams x %lld per position)\n", "layers dumped",
                    o.dump_layers.c_str(), (long long) g.n_layers, (int) g.hc, (long long) g.n_embd);
    }
    if (half_dump != nullptr) {
        std::fclose(half_dump);
        strata::gpu::free(half_stage);
        std::printf("%-24s %s (%lld layers x %llu per position)\n", "halves dumped", o.dump_halves.c_str(),
                    (long long) g.n_layers, (unsigned long long) half_stride);
    }
    if (routing != nullptr) {
        std::fclose(routing);
        drive.routing = nullptr;
        std::printf("%-24s %s (%lld records of layer, k, ids, weights)\n", "routing dumped",
                    o.dump_routing.c_str(), (long long) drive.calls);
    }
    if (o.stage_timing) strata::core::stage_timing_report(g.n_layers);

    const int64_t decoded = (int64_t) produced.size();
    std::printf("prompt  :");
    for (int64_t t : o.tokens) std::printf(" %lld", (long long) t);
    std::printf("\noutput  :");
    for (int64_t t : produced) std::printf(" %lld", (long long) t);
    std::printf("\n");
    const double decode_ms = decoded > 0 ? total_ms / (double) decoded : 0.0;
    std::printf("%-24s %lld tokens in %.1f ms  ->  %.2f tok/s\n", "decode", (long long) decoded, total_ms,
                decode_ms > 0.0 ? 1000.0 / decode_ms : 0.0);
    if (n_prompt > 1)
        std::printf("%-24s %lld tokens in %.1f ms  ->  %.2f tok/s  (time to first token %.1f ms)\n", "prefill",
                    (long long) (n_prompt - 1), prefill_ms,
                    prefill_ms > 0 ? 1000.0 * (double) (n_prompt - 1) / prefill_ms : 0.0, ttft_ms);
    if (!o.dump_mixed.empty()) {
        std::vector<float> mx((size_t) g.n_embd);
        if (!strata::gpu::copy(mx.data(), ss.block.mixed, mx.size() * sizeof(float))) {
            std::fprintf(stderr, "strata generate: reading mixed back failed\n");
            return 1;
        }
        std::FILE* mf = std::fopen(o.dump_mixed.c_str(), "wb");
        if (mf == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_mixed.c_str());
            return 1;
        }
        std::fwrite(mx.data(), sizeof(float), mx.size(), mf);
        std::fclose(mf);
        double s2 = 0, mag = 0;
        for (float v : mx) { s2 += (double) v * (double) v; mag += std::fabs((double) v); }
        std::printf("%-24s %s (n_embd %lld, rms %.5g, mean|.| %.5g)\n", "mixed dumped", o.dump_mixed.c_str(),
                    (long long) g.n_embd, std::sqrt(s2 / (double) mx.size()), mag / (double) mx.size());
    }

    // ---- the residual, for bisecting the head against the layers (see `dump_residual`'s note)
    if (!o.dump_residual.empty()) {
        std::vector<float> R((size_t) g.hc * g.n_embd);
        if (!strata::gpu::copy(R.data(), ss.R, R.size() * sizeof(float))) {
            std::fprintf(stderr, "strata generate: reading R back failed\n");
            return 1;
        }
        std::FILE* rf = std::fopen(o.dump_residual.c_str(), "wb");
        if (rf == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_residual.c_str());
            return 1;
        }
        const int32_t hdr[2] = {(int32_t) g.hc, (int32_t) g.n_embd};
        std::fwrite(hdr, sizeof hdr, 1, rf);
        std::fwrite(R.data(), sizeof(float), R.size(), rf);
        std::fclose(rf);
        double mag = 0, mx = 0;
        int bad = 0;
        for (float v : R) {
            if (!std::isfinite(v)) ++bad;
            else { mag += std::fabs((double) v); mx = std::max(mx, (double) std::fabs((double) v)); }
        }
        std::printf("%-24s %s (%d x %lld, nonfinite %d, mean|.| %.4g, max|.| %.4g)\n", "residual dumped",
                    o.dump_residual.c_str(), (int) g.hc, (long long) g.n_embd, bad, mag / (double) R.size(), mx);
    }

    if (o.stats) {
        std::printf("%-24s %.3f ms/token (WALL CLOCK: embed, layers, head, sample)\n", "  per token", decode_ms);
        // **THE PER-TOKEN HOST TERM, WHICH `--gpu-only-full` CANNOT SEE.**  That measurement never enters the
        // token loop, so it excludes all six of these.  On the 78-token fixture + 200 generated tokens the six
        // sum to ~9 ms of non-layer work against ~1.5 ms of actual head GPU work - 16% of the token, and it is
        // not the pool.
        if (phase_tokens > 0) {
            const double pt = (double) phase_tokens;
            std::printf("%-24s PLE %.3f  embed %.3f  LAYERS %.3f  head %.3f  readback %.3f  sample %.3f  "
                        "(sum %.3f of %.3f ms)\n",
                        "  token host phases", ms_ple / pt, ms_embed / pt, ms_layers / pt, ms_head / pt,
                        ms_readback / pt, ms_sample / pt,
                        (ms_ple + ms_embed + ms_layers + ms_head + ms_readback + ms_sample) / pt, decode_ms);
        }
        if (const std::string io = ple_table.io_report(); !io.empty()) std::printf("  %s\n", io.c_str());
        // **THE DENOMINATOR IS THE POSITIONS THE POOL ACTUALLY RAN ON, NOT THE DECODED TOKENS (A6).**
        // `drive_pool` is called once per layer per position and PREFILL runs the loop too, so accumulating
        // `cpu_ms` over prefill and then dividing by `decoded` inflates this figure.  `drive.calls / n_layers`
        // is the number of positions - the same correction the ring counters below already received, which is
        // why they print "of 192" rather than "240 of 192".
        const double pool_positions = g.n_layers > 0 ? (double) drive.calls / (double) g.n_layers : 0.0;
        std::printf("%-24s %.3f ms/token over %lld layers (%.0f positions, %lld dispatches)\n",
                    "  the CPU expert pool", pool_positions > 0.0 ? drive.cpu_ms / pool_positions : 0.0,
                    (long long) g.n_layers, pool_positions, (long long) drive.calls);
        // **AND WHERE INSIDE `run()` IT WENT.**  Three phases per layer and they were one number, which cannot
        // tell a pool that is slow at the WORK from one that is slow at the SYNCHRONISATION - opposite fixes.
        // Wait-for-park is expected to be ~0 (the workers re-parked at the end of the previous layer); the
        // question is whether the time is in the drain or in the re-park barrier.
        if (pool_positions > 0.0) {
            double wp = 0, dr = 0, rp = 0;
            pool.phase_ms(wp, dr, rp);
            const double per = pool_positions;
            std::printf("%-24s   wait-park %.3f  drain %.3f  re-park %.3f  ms/token\n",
                        "  pool phases", wp / per, dr / per, rp / per);
        }
        std::printf("%-24s %lld blobs read\n", "  expert blobs", (long long) srcp->reads());
        for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0)
            std::printf("  CUDA%d experts           %lld routed entries computed\n",
                        r + 1, (long long) remote_experts[(size_t) r].computed());
        // ---- **R4's DISPATCH MEASUREMENT: h, ON THE ENGINE'S OWN ROUTING.**  No offline trace, no corpus
        // question, no k-fold - these are the ids the router actually produced on this run.  Reported as
        // hits/lookups so it can be read directly as the h the cache would deliver, and alongside `refused`
        // so a full cache is visible rather than silently capping the rate.
        if (o.expert_cache > 0) {
            const int64_t look = drive.d.cache_hits + drive.d.cache_admitted + drive.d.cache_refused;
            const int64_t hl = drive.d.hit_ready + drive.d.hit_late;
            std::printf("%-24s %lld of %lld layers the hit work was DONE when the pool returned\n",
                        "  R4 overlap", (long long) drive.d.hit_ready, (long long) hl);
            std::printf("%-24s %lld of %lld = %.4f      (%lld admitted, %lld refused, cache %.4f%% full)\n",
                        "  R4 expert-cache hits", (long long) drive.d.cache_hits, (long long) look,
                        look > 0 ? (double) drive.d.cache_hits / (double) look : 0.0,
                        (long long) drive.d.cache_admitted, (long long) drive.d.cache_refused,
                        100.0 * (double) (drive.d.cache_admitted + drive.d.cache_hits > 0
                                              ? (double) xcache.resident() / (double) xcache.slots()
                                              : 0.0));
        }
        if (tgraph.captured && tgraph.calls > 0) {
            const double per = (double) tgraph.calls;
            std::printf("%-24s wait for rings %.3f  pool %.3f ms/token  (%lld flushes over %lld positions)\n",
                        "  token graph", tgraph.ms_wait / per, tgraph.ms_pool / per, (long long) tgraph.flushes,
                        (long long) tgraph.calls);
        }
        if (gr.captured && gr.calls_total > 0) {
            // The counters are CUMULATIVE over every `session_loop` call, and PREFILL runs the loop too - so
            // the denominator is the number of positions, not the number of generated tokens.  Dividing by
            // `n_layers * decoded` printed "240 of 192", which is a reporting bug that looks like a ring
            // firing more often than it should.
            const int64_t positions = gr.calls_total;
            std::printf("%-24s %lld of %lld over %lld positions\n", "  rings seen MID-GRAPH",
                        (long long) gr.rings_mid_graph, (long long) (g.n_layers * positions),
                        (long long) positions);
            std::printf("%-24s %.3f ms of a %.3f ms layer\n", "  ring latency",
                        gr.ms_to_ring / (double) (g.n_layers * positions), decode_ms / (double) g.n_layers);
            // **THE ROUND TRIP, SPLIT AT THE RING.**  `ring latency` is the first half and stops when the ring
            // is seen; this is the second half - the driver calls after it, during which the GPU is IDLE
            // because `post[l]` has not been launched yet.  `--no-pool` is the arm that isolates it: 38.73
            // ms/token against a 26.32 ms pure-GPU floor is 12.4 ms of round trip with no expert work at all.
            //
            // Same denominator as the pool line above (the positions the loop actually ran on), so the two can
            // be added without one of them being inflated by prefill.
            const double perlap = (double) (g.n_layers * positions);
            std::printf("%-24s %.3f ms/token over %.0f positions (%.3f ms/layer, after the ring)\n",
                        "  host after ring", pool_positions > 0.0 ? gr.ms_host / pool_positions : 0.0,
                        pool_positions, gr.ms_host / perlap);
        }
    }

    if (dump != nullptr) std::printf("%-24s %s\n", "logits dumped", o.dump_logits.c_str());

    strata::core::session_graphs_free(gr);
    strata::core::doorbell_free(db);
    strata::gpu::free(d_next);
    strata::gpu::free(d_logits);
    strata::gpu::free(d_emb);
    strata::gpu::free(d_parts);
    strata::core::session_release(ss);
    strata::gpu::free(sbuf);
    strata::gpu::free(arena);
    return 0;
}
