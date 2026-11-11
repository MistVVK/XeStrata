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
    /// A DoRA's row norms: the slot's weight's, one a row of it (its `w->n` rows), from the file's tensors.
    std::function<bool(const DitLoraSlot&, std::vector<float>&, std::string&)> dora_norms;
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

    /// The table: cleared, and a weight's slot added (a weight with no rows, or rows past it, is none).
    void clear_slots() { slots_.clear(); }
    void add(std::string name, DitW& w, int64_t row0, int64_t rows, int64_t src_row = 0, bool biased = false);
    void add(std::string name, DitW& w, bool biased) { add(std::move(name), w, 0, w.n, 0, biased); }
    std::vector<DitLoraTarget> targets() const;

    /// The adapters the next steps apply, each with its strength, bound to the weights' products: a key the DiT does
    /// not take (or a shape that is not its weight's, or a form the engine does not apply) is false with err, and
    /// nothing bound.  The same set again (its files and strengths) changes nothing; `changed`: it was another.
    bool set(const std::vector<LoraUse>& loras, std::string& err, bool& changed);
    /// The set let go: the weights' layers, the bake, the buffers.
    void drop();

    const std::string& key() const { return key_; }
    int count() const { return (int) files_.size(); }
    /// The bound entries' A and B on the GPU (and a GLoRA's buffers).
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
};

/// A slot's row norms out of its weight's tensor's (`norms`: the tensor's rows, the slot's from src_row on).
bool dit_slot_norms(const DitLoraSlot& slot, const std::vector<float>& norms, std::vector<float>& out,
                    std::string& err, const char* what);
/// The row norms of a bf16 tensor [rows][k], FP32 (summed in double).
std::vector<float> dit_bf16_row_norms(const uint16_t* h, int64_t rows, int64_t k);

}  // namespace strata::image
