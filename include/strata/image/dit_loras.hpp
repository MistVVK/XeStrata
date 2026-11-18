// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/dit_loras.hpp - the adapters a DiT binds to its weights' products, one binder for every image
// model's DiT (Qwen-Image, Mage-Flow, MOSS-Image, Anima): the table of the weights a file's keys land on (each DiT
// fills it from its names, lora_names.hpp), a request's set matched against it and checked, its entries uploaded and
// bound to the weights (image_lora.hpp's LoraLayer), the base products' DoRA factors, a GLoRA's buffers, and the
// blocks' plain LoRAs folded into their weights (bake).  What differs between the DiTs comes through DitLoraHooks.
#pragma once

#include "strata/image/dit_product.hpp"
#include "strata/image/image_lora.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace strata::image {

/// A weight and the rows of a file's entry it takes: the weight's rows [row0, row0 + rows) are the entry's rows from
/// src_row on (a fused weight is several slots of one name, a weight split on the GPU several of one entry).  biased:
/// the weight's product adds a bias (an entry's bias's delta is added beside it).
struct DitLoraSlot {
    std::string name;
    DitW* w = nullptr;
    int64_t row0 = 0, rows = 0, src_row = 0;
    bool biased = false;
};

/// A slot as the probe and the server see it: the weight's name, its rows and columns, and the entry's rows it takes.
struct DitLoraTarget {
    std::string name;
    int64_t row0 = 0, rows = 0, cols = 0, src_row = 0;
};

/// What a DiT does its own way.
struct DitLoraHooks {
    /// The DiT's drop_loras: the set let go (DitLoras::drop) and the DiT's state made from it.
    std::function<void()> drop;
    /// A DoRA's composed row norms (PEFT's ||W + (alpha/rank) BA||, dit_dora_norms): the slot's rows, one a row,
    /// from the weight's tensors and the entry's factors.
    std::function<bool(const DitLoraSlot&, const LoraEntry&, std::vector<float>&, std::string&)> dora_norms;
    /// An entry the DiT does not take on the slot: true, and why (null: none refused).
    std::function<bool(const DitLoraSlot&, const LoraEntry&, std::string&)> refuse;
    /// An entry's key as the DiT reads it, where it differs: true and the key (null: as written); `file` is called
    /// before a file's entries.
    std::function<void(const LoraFile&)> file;
    std::function<bool(const LoraEntry&, std::string&)> rename;
    /// The blocks' weights (their copies are prepared as they reach the GPU: bake folds plain LoRAs in there), and
    /// the blocks made again by that prepare (the stream reloaded).
    std::function<void(std::vector<DitW*>&)> block_weights;
    std::function<void()> reprepare;
};

class DitLoras {
public:
    /// The deltas' buffers (the DiT's products'), the DiT's allocations and stream (the members' addresses), and its
    /// name for the messages.
    DitLoras(DitLoraScratch* scratch, std::vector<void*>* owned, void* const* stream, const char* what)
        : scratch_(scratch), owned_(owned), s_(stream), what_(what) {}
    DitLoras(const DitLoras&) = delete;
    DitLoras& operator=(const DitLoras&) = delete;

    DitLoraHooks hooks;
    /// The blocks' plain LoRAs folded into their weights (the default), else added to each product.
    bool bake = true;
    /// The path a block's weights' names start with, its number after it ("transformer_blocks.", "layers.",
    /// "blocks."): what a LoraUse's block factors are read by.
    std::string block_path;

    /// The table: cleared, and a weight's slot added (a weight with no rows, or rows past it, is none).
    void clear_slots() { slots_.clear(); }
    void add(std::string name, DitW& w, int64_t row0, int64_t rows, int64_t src_row = 0, bool biased = false);
    void add(std::string name, DitW& w, bool biased) { add(std::move(name), w, 0, w.n, 0, biased); }
    std::vector<DitLoraTarget> targets() const;
    /// The block a weight's name is in (block_path's number), -1 outside the blocks; and the blocks the table has.
    int block_of(const std::string& name) const;
    int blocks() const;

    /// The adapters the next steps apply, each with its strength, bound to the weights' products: a key the DiT does
    /// not take (or a shape that is not its weight's, or a form the engine does not apply) is false with err, and
    /// nothing bound.  The same set again (its files and strengths) changes nothing; `changed`: it was another.
    bool set(const std::vector<LoraUse>& loras, std::string& err, bool& changed);
    /// The set let go: the weights' layers, the bake, the buffers.
    void drop();

    const std::string& key() const { return key_; }
    int count() const { return (int) files_.size(); }
    /// The bound entries' A and B on the GPU (and a GLoRA's buffers, and the bake's: bake_tmp and bake_at16).
    uint64_t bytes() const;
    bool baked_any() const { return baked_any_; }
    float* bake_tmp() const { return bake_tmp_; }        // a weight's delta [n][k], while adapters are baked
    uint16_t* bake_at16() const { return bake_at16_; }   // an adapter's A^T

private:
    DitLoraScratch* scratch_;
    std::vector<void*>* owned_;
    void* const* s_;
    const char* what_;
    std::vector<DitLoraSlot> slots_;
    std::vector<std::unique_ptr<LoraLayer>> layers_;
    std::vector<std::shared_ptr<const LoraFile>> files_;
    std::string key_;
    bool baked_any_ = false;
    float* bake_tmp_ = nullptr;
    uint16_t* bake_at16_ = nullptr;
    uint64_t bake_bytes_ = 0;   // bake_tmp_'s and bake_at16_'s
};

/// A weight tensor's rows as FP32: read(r0, n, out, err) fills rows [r0, r0 + n) of it, k values a row.  The DiT's
/// own storage behind it (a file's bf16 rows, a GGUF's quantized blocks dequantized as the products read them).
using DitTensorRows = std::function<bool(int64_t, int64_t, float*, std::string&)>;
/// A bf16 tensor's rows ([.][k] host bits) as a DitTensorRows.
DitTensorRows dit_bf16_rows(const uint16_t* h, int64_t k);
/// A DoRA's row norms as PEFT takes them: ||W_r + s (BA)_r|| with s = alpha / rank (the entry's own scale, not the
/// request's strength), over the slot's `rows` rows - the weight tensor's rows [trow0, trow0 + rows) against the
/// entry's rows [src_row, src_row + rows).  The composed weight is never made:
///   ||W_r + s (BA)_r||^2 = |W_r|^2 + 2 s sum_j B_rj (W A^T)_rj + s^2 B_r (A A^T) B_r^T,
/// W A^T and A A^T one pass each while the weight's rows are read (a LoRA's bind, not a generation step).
bool dit_dora_norms(const LoraEntry& e, int64_t src_row, int64_t trow0, int64_t rows, int64_t k,
                    const DitTensorRows& read, std::vector<float>& norms, std::string& err);

}  // namespace strata::image
