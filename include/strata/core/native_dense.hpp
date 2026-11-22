// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2026 hiruhiru2005
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
#pragma once

#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace strata::core {
class WeightTable;
struct WeightRef;

/// A packed dense replacement for a native GGUF matrix. Its bytes stay in dense_nq.bin until read.
struct DenseNqEntry {
    int type = -1;
    int64_t ne0 = 0, ne1 = 0;
    uint64_t offset = 0, bytes = 0;
};
bool dense_nq_init(const std::string& pack_dir, std::string& err);
bool dense_nq_active();
const DenseNqEntry* dense_nq_find(const std::string& name);
bool dense_nq_read(const std::string& name, int64_t ne0, int64_t ne1, std::vector<uint8_t>& data,
                   std::string& err);

/// STRATA_HC_Q8 with the BF16 copies dropped (NativeDense::drop_hc_bf16): `w`'s BF16 values, its Q8_0 copy
/// dequantized on `stream` into scratch `slot` (0-1 the prompt path's down and up, 2-3 the single-token read's),
/// which `drop_hc_bf16` allocated; `w->data` when it is resident.  A slot holds one matrix at a time: its reader
/// is done with it before the next view of the same slot is queued on the same stream.
const uint16_t* hc_bf16_view(const WeightRef* w, int slot, void* stream);

// Experimental GDN/QSA/shared-expert projection overrides. Upload unchanged native GGUF
// blocks once, then attach them to the matching canonical WeightRef. Unsupported
// types retain their canonical paths. Owns one Q8_1 scratch vector shared by all
// these projections, so use one ordered session stream and keep this object
// alive until all graphs that reference it have been destroyed and synchronized.
class NativeDense {
public:
    NativeDense() = default;
    ~NativeDense();
    NativeDense(const NativeDense&) = delete;
    NativeDense& operator=(const NativeDense&) = delete;
    bool load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
              bool include_ple_key = false);
    /// Plan v0.3 P1: the canonical tensor names `load` would serve natively from these shards (eligible name,
    /// supported type, 2-D), read from the GGUF headers only - so the canonical arena can skip them.
    static bool served_names(const std::vector<std::string>& shards, bool include_ple_key,
                             std::set<std::string>& out, std::string& err);
    /// Layer split: load only blocks [lb, le) (every other `blk.N.` projection belongs to another GPU's stage; the
    /// PLE tensors are loaded everywhere).  Process-wide, read by the next `load`; (-1, -1) = all layers.
    static void set_layer_range(int lb, int le);
    /// The bytes `load` would upload for each layer (`per_layer[l]`), and for the tensors every stage loads (the PLE
    /// ones: `shared`), from the GGUF headers only - so a layer split can price a placement before loading anything.
    static bool layer_bytes(const std::vector<std::string>& shards, bool include_ple_key, int64_t n_layers,
                            std::vector<uint64_t>& per_layer, uint64_t& shared, std::string& err);
    /// #326: a native pack whose `blk.1.ple_key.weight` row is unquantized (iq_pack --compat-bf16 of a GGUF key
    /// the native kernel also reads, e.g. OrcaRouter's IQ3_XXS) serves the PLE from that row, so it is taken out
    /// of `skip` and `load` does not upload the GGUF key over it.  A quantized row leaves `skip` unchanged.
    static bool keep_unquantized_ple_key(const std::string& pack_dir, std::set<std::string>& skip, std::string& err);
    /// STRATA_HC_Q8 (not 0): the layers' hyper-connection down / up projections can be read from Q8_0 copies, so
    /// the canonical arena may leave their BF16 out (`hc_side_names`) and load it beside, to be dropped after `load`.
    static bool hc_q8_wanted();
    /// The pack's `blk.N.hc_{attn,ffn}_{down,up}.weight` rows (`hc`) and every other row (`others`), from its index.
    static bool hc_side_names(const std::string& pack_dir, std::set<std::string>& hc, std::set<std::string>& others,
                              std::string& err);
    /// Points `main`'s rows at `side`'s loaded copies of them (the BF16 the canonical arena left out).
    static void adopt(WeightTable& main, const WeightTable& side);
    /// The next `load` may give the BF16 copies up (they are in a side allocation the caller frees after
    /// `drop_hc_bf16`): no VRAM-share rule for the choice, since the Q8_0 copies then take less than they free.
    void set_hc_side(bool on) { hc_side_ = on; }
    /// After `load`: the Q8_0 read was chosen.
    bool hc_q8_chosen() const { return !hc_q8_refs_.empty(); }
    /// The Q8_0 hyper-connection projections' VRAM (0 once they are in pinned memory).
    uint64_t hc_q8_device_bytes() const { return hc_q8_host_ ? 0 : hc_q8_bytes_; }
    /// Moves the Q8_0 hyper-connection projections to pinned host memory, the kernels reading them there in place
    /// (a card the expert cache would leave short otherwise).  Before any graph captures them.
    bool hc_q8_to_host(WeightTable& table, std::string& err);
    /// The blocks the layers' projections are allocated in (`load`), some layers each: the last step for a card the
    /// expert cache would leave short moves the fewest of them to pinned host memory, the kernels reading them there in
    /// place (a window reads them whole over the link).  Before any graph captures them.
    size_t layer_blocks() const { return blocks_.size(); }
    /// Block `i`'s VRAM (0 once in pinned memory, or a block of rows outside the layers), and its first and last layer.
    uint64_t block_device_bytes(size_t i) const {
        return blocks_[i].p != nullptr && blocks_[i].l0 >= 0 ? blocks_[i].used : 0;
    }
    int block_first_layer(size_t i) const { return blocks_[i].l0; }
    int block_last_layer(size_t i) const { return blocks_[i].l1; }
    bool block_to_host(WeightTable& table, size_t i, std::string& err);
    /// The layers' BF16 down / up rows that have Q8_0 copies become non-resident (data null) and the dequantization
    /// scratch hc_bf16_view uses is allocated here, at load, not on first use (a graph may capture the first use).
    bool drop_hc_bf16(WeightTable& table, std::string& err);
    uint64_t weight_bytes() const { return bytes_; }
    size_t tensor_count() const { return count_; }

    /// A device allocation projections are carved from (layers l0..l1; -1: rows outside the layers)
    struct Block {
        void* p = nullptr;
        uint64_t cap = 0, used = 0;
        int l0 = -1, l1 = -1;
    };

private:
    std::vector<Block> blocks_;            ///< the projections'
    std::vector<Block> hc_blocks_;         ///< the Q8_0 hyper-connection copies'
    std::vector<void*> host_;              ///< the pinned copies of moved blocks
    size_t count_ = 0;
    std::vector<WeightRef*> hc_q8_refs_;   ///< the rows with a Q8_0 copy
    uint64_t hc_q8_bytes_ = 0;
    bool hc_q8_host_ = false;
    void* scratch_ = nullptr;
    uint64_t bytes_ = 0;
    bool hc_side_ = false;
};
} // namespace strata::core
