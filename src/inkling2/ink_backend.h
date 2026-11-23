// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/inkling2/ink_backend.h - Inkling-2-Mini's backend interface (upstream's, of which XeStrata has the GPU one)
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ink_batch.h"
#include "ink_manifest.h"

namespace ink2 {

struct ImageRec;  // defined in ink_vision.h

struct BackendOpts {
    int max_ctx = 32768;
    int parallel = 1;            // 1..64 concurrent slots
    int chunk = 512;
    int gpu_index = 0;
    int vram_reserve_mib = -1;   // >= 0 reserves N MiB free VRAM & activates hybrid offload (#6520)
    int threads = 0;             // 0 = auto
    double vram_experts_gb = -1.0; // <0 = auto (3x loop-weighted vram_load_order!)
    double ram_experts_gb = -1.0;  // <0 = auto
    KvDtype kv_dtype = KvDtype::FP16;
    bool kv_dtype_auto = true;   // auto selects FP8_E4M3 when parallel >= 16 on <= 32 GiB GPU
    bool enable_dspark = true;
    std::string vision_path;
    // The QSA indexer (QSAW v1, sycl/ink_sy_qsa.h): loaded where a path is given; a slot uses it where its request
    // says so (serve's qsa= key, set_slot_qsa), else `qsa_default`.  The tokens a query keeps: `qsa_budget` in all,
    // the first `qsa_sink` and the last `qsa_local` of them always (multiples of the indexer's 4-token block).
    // PROVISIONAL (2026-11-23): 2048 / 256 / 16 are the fork's engine's defaults (inkling2_engine.cu's
    // QsaIndexerWeights) and its README's recall table's budget; the MODEL check's measurements decide them.
    std::string qsa_path;
    bool qsa_default = false;
    int qsa_budget = 2048;
    int qsa_local = 256;
    int qsa_sink = 16;
};

/// Whether the QSA settings are whole blocks and leave the indexer room to choose (`err` says which is not).
inline bool qsa_geometry_ok(const BackendOpts& o, std::string& err) {
    if (o.qsa_budget % 4 || o.qsa_local % 4 || o.qsa_sink % 4) {
        err = "--qsa-budget, --qsa-local and --qsa-sink take multiples of 4 (the indexer's block)";
        return false;
    }
    if (o.qsa_sink + o.qsa_local >= o.qsa_budget) {
        err = "--qsa-budget must exceed --qsa-sink + --qsa-local";
        return false;
    }
    return true;
}

struct BackendStats {
    size_t vram_experts_loaded = 0;
    size_t ram_experts_loaded = 0;
    size_t total_experts = 0;
    size_t vram_used_bytes = 0;
    size_t vram_total_bytes = 0;
    int active_slots = 0;
    int max_slots = 1;
    KvDtype kv_dtype = KvDtype::FP16;
};

class InkBackend {
public:
    virtual ~InkBackend() = default;

    virtual const char* backend_name() const noexcept = 0;
    virtual std::string device_info() const = 0;
    virtual BackendStats stats() const = 0;

    // Load Section 1 trunk + DSpark + 3x loop-weighted Section 4 experts into device/host memory
    virtual bool init(const InkModelFile& model, const BackendOpts& opts, std::string& err) = 0;

    // Slot lifecycle (0 <= slot < opts.parallel)
    virtual bool alloc_slot(int slot, int max_ctx, KvDtype kv_dtype, std::string& err) = 0;
    virtual void free_slot(int slot) = 0;
    virtual void reset_slot(int slot) = 0;
    // Speculative rollback: restore KV + ShortConv state to `keep_len` (requires snapshot taken before verify!)
    virtual void snapshot_slot(int slot) = 0;
    virtual void rollback_slot(int slot, int keep_len) = 0;

    // Batched 48-visit forward across 1..64 active slots!
    // logits_out: [batch.count_logits(), vocab_unpadded] FP32 (already softcapped by 30*tanh(z/30) and sliced to 200058).
    virtual bool forward_batch(
        const StepBatch& batch,
        float* logits_out,
        TapCapture* taps_out,
        std::string& err) = 0;

    // On-device DSpark Block-8 Recurrent Markov drafting for one slot.
    // Returns number of drafted tokens (0..max_draft, typically up to 7); returns 0 if unavailable/disabled.
    virtual int dspark_draft(
        int slot,
        int32_t anchor_tok,
        int anchor_pos,
        int max_draft,
        int32_t* out_draft_ids,
        std::string& err) {
        (void)slot; (void)anchor_tok; (void)anchor_pos; (void)max_draft; (void)out_draft_ids; (void)err;
        return 0;
    }

    // Optional fused on-device greedy speculative step (W=4 Adaptive Beam Tree + zero-logits-D2H verify + commit).
    // Returns n_emit in 1..min(8, max_emit) on success, 0 to fall back to dspark_draft + forward_batch, or -1 on error.
    virtual int dspark_spec_step_greedy(
        int slot,
        int32_t anchor_tok,
        int anchor_pos,
        int max_emit,
        int32_t* out_emitted,
        int* out_drafted,
        std::string& err) {
        (void)slot; (void)anchor_tok; (void)anchor_pos; (void)max_emit;
        (void)out_emitted; (void)out_drafted; (void)err;
        return 0;
    }

    // Optional GPU Vision ViT encoder: returns true if encoded on device, false to fall back to CPU ViT.
    virtual bool encode_image(const ImageRec& img, std::vector<float>& out_embeds, std::string& err) {
        (void)img; (void)out_embeds; (void)err;
        return false;
    }

    // Whether `slot`'s queries attend to the blocks the QSA indexer picks (where one is loaded) or to every cell
    virtual void set_slot_qsa(int slot, bool on) { (void)slot; (void)on; }

    // Verify Section 5 selfcheck / golden ops parity on this backend
    virtual bool selfcheck(std::string& err) = 0;
};

// The GPU backend (sycl/ink_sycl_backend.cpp): the engine's one, on the GPU XeStrata selected.  Upstream's CPU,
// CUDA, HIP, Metal and dual-GPU backends are not taken (XeStrata runs NVIDIA's and AMD's GPUs through the same SYCL
// kernels).
std::unique_ptr<InkBackend> make_sycl_backend();

}  // namespace ink2
