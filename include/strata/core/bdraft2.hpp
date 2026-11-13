// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// The model-specific YASAKANI block drafter's weights and output head.
#pragma once

#include "strata/artifact/yskn.hpp"
#include "strata/core/shead.hpp"
#include "strata/kernels/yskn.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::core {

class Bdraft2Weights {
public:
    Bdraft2Weights() = default;
    ~Bdraft2Weights();
    Bdraft2Weights(const Bdraft2Weights&) = delete;
    Bdraft2Weights& operator=(const Bdraft2Weights&) = delete;
    // Validate the draft's complete shape against its target before allocating device memory.
    bool load(const std::string& path, const std::string& head, int device, int target_hidden, int target_layers,
              int target_vocab, std::string& err);
    // Caller drains all streams first. A failed load leaves no live allocations.
    void release();
    bool loaded() const { return container_ != nullptr; }
    const strata::yskn::ModelDesc& model() const { return container_->model(); }
    const strata::kernels::YsknDev& matrix(const std::string& name) const { return matrices_.at(name); }
    const void* vector(const std::string& name) const { return vectors_.at(name); }
    // scratch is at least scratch_bytes(); x and y are contiguous FP32 rows on the same device.
    void project(const std::string& name, const float* x, int rows, float* y, void* scratch, void* stream) const;
    size_t scratch_bytes() const { return scratch_bytes_; }
    struct ContextWorkspace {
        float *fc = nullptr, *normalized = nullptr, *key = nullptr, *value = nullptr;
        int8_t* quantized = nullptr;   ///< [8][hidden], the rows the shared ring log saves
        float* scales = nullptr;      ///< [8][hidden / 512]
        void* projection = nullptr;
    };
    // Commit 1..8 rows of five concatenated target taps into each layer's ring. positions and rope are device arrays;
    // rope is [rows][2][64] from df2_rope_table_cpu. ring arrays are each [8][slots][128], and tag/epoch can be null.
    void context(const float* taps, int rows, const int32_t* positions, const float* rope, float* const* keys,
                 float* const* values, int slots, int64_t* tag, const int32_t* epoch, const ContextWorkspace& work,
                 void* stream) const;
    // Rebuild the ring from rows saved by the shared draft ring log. The inputs have the same device layout as
    // ContextWorkspace::quantized and scales; their bytes are exactly those used for the original update.
    void context_replay(const int8_t* quantized, const float* scales, int rows, const int32_t* positions,
                        const float* rope, float* const* keys, float* const* values, int slots, int64_t* tag,
                        const int32_t* epoch, const ContextWorkspace& work, void* stream) const;
    struct BlockWorkspace {
        // Eight rows: h, normalized, conv and output are [8][5120]; dynamic is [8][1280].
        float *h = nullptr, *normalized = nullptr, *conv = nullptr, *dynamic = nullptr, *output = nullptr;
        // q/qnorm are [8][4096]; k/knorm/v are [8][1024]; attention is [8][4096].
        float *q = nullptr, *qnorm = nullptr, *k = nullptr, *knorm = nullptr, *v = nullptr, *attention = nullptr;
        // gate/up/glu are [8][17408]; final is [7][5120]; hp is [7][256].
        float *gate = nullptr, *up = nullptr, *glu = nullptr, *final = nullptr, *hp = nullptr;
        // part is [33][8][32][130]; ids and scores are [7][16].
        float *part = nullptr, *scores = nullptr;
        int32_t* ids = nullptr;
        void* projection = nullptr;
    };
    // One eight-row block. Embedding rows and all other arguments are device buffers. The ring holds committed
    // context only; this block never writes it. The caller owns and sizes every workspace buffer.
    void block(const float* anchor_embedding, const float* mask_embedding, const int32_t* anchor,
               const int32_t* anchor_pos, const float* rope, float* const* keys, float* const* values, int slots,
               const int64_t* tag, const int32_t* epoch, const BlockWorkspace& work, int32_t* out,
               float* probabilities, void* stream);
    // Final norm output [rows][hidden] to token IDs and scores [rows][16], on stream.
    void top512(const float* h, int rows, int32_t* ids, float* values, void* stream);
    // The full vocabulary head is owned by the target. projection is at least yskn_scratch_bytes(*weights, 7),
    // logits is [7][vocab], and top_scratch is [7][kDf2TopBlocks][kDf2TopK][2]. All buffers stay on device.
    struct FullHead {
        const strata::kernels::YsknDev* weights = nullptr;
        void* projection = nullptr;
        float* logits = nullptr;
        uint32_t* top_scratch = nullptr;
    };
    void set_full_head(const FullHead& full) { full_head_ = full; }
    enum class HeadMode { Auto, Off, On };
    // Auto starts with the full head. The startup probe tests Off and On and applies its verdict to Auto.
    // Off uses the full target head. On requires the model-specific SHEAD. Changes must precede graph capture.
    void set_head_mode(HeadMode mode) { head_mode_ = mode; }
    void set_probe_shead(bool use) { probe_shead_ = use; }
    bool has_shead() const { return head_.loaded(); }
    bool uses_shead() const { return head_mode_ == HeadMode::On || (head_mode_ == HeadMode::Auto && probe_shead_); }
    // The selector projects the normalized rows, scores the selected head's candidates, and walks from anchor.
    // hp is [rows][rank], ids and values are [rows][16]; all pointers are device buffers on stream.
    void select(const float* h, int rows, const int32_t* anchor, float* hp, int32_t* ids, float* values,
                int32_t* out, float* probabilities, void* stream);
    uint64_t bytes() const { return bytes_ + head_.bytes(); }

private:
    int device_ = -1;
    uint64_t bytes_ = 0;
    size_t scratch_bytes_ = 0;
    std::unique_ptr<strata::yskn::Container> container_;
    std::unordered_map<std::string, strata::kernels::YsknDev> matrices_;
    std::unordered_map<std::string, const void*> vectors_;
    std::vector<void*> allocations_;
    Shead head_;
    FullHead full_head_;
    HeadMode head_mode_ = HeadMode::Auto;
    bool probe_shead_ = false;
};

}  // namespace strata::core
