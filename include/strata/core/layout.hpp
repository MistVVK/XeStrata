// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/core/layout.hpp - the pack's LAYOUT, resolved by name and checked against the kernels.
//
// `WeightTable` answers "where are the bytes".  This answers "is this the tensor I think it is", and it is
// the layer where a pack and a kernel disagree - which is the failure this project keeps paying for.  Two
// examples, both real:
//
//   * round 194 sized the indexer key store from `indexer.head_count = 4`, which counts QUERY heads.  The
//     cached key is ONE shared head of 128 (`indexer.k_proj` is [2560, 128]), so the term was 4x too big and
//     the error survived a round because it moved in the direction that TIGHTENS the budget.
//   * `docs/semantics.md` and the tensor manifest agree on every dimension here, but nothing CHECKED that a
//     named tensor had the shape the kernel reading it assumes.
//
// So this header does two things: it names the tensors per layer type, and it asserts every shape the
// kernels depend on.  A mismatch is reported with the tensor name, the shape found and the shape required,
// at LOAD time - not as a wrong number inside a GEMV at token 4000.
#pragma once

#include "strata/core/weights.hpp"

#include <cstdint>
#include <string>

namespace strata {
struct ModelSpec;
}

namespace strata::core {

/// The model's geometry, taken from `docs/semantics.md` and the artifact's own metadata.  Every field here
/// is a number a kernel depends on, so a change is a change to a kernel contract and not a tuning knob.
struct ModelGeometry {
    int64_t n_embd = 2560;
    int64_t n_layers = 48;
    int64_t qsa_interval = 4;      ///< every 4th layer is full attention: layers 3, 7, ... 47

    // GDN (36 layers)
    int64_t ssm_state_size = 128;
    int64_t ssm_k_heads = 16;
    int64_t ssm_v_heads = 48;
    int64_t ssm_d_conv = 4;
    int64_t ssm_conv_channels = 10240;   ///< 2*128*16 + 128*48
    int64_t ssm_value_dim = 6144;        ///< 128 * 48

    // QSA (12 layers)
    int64_t n_head = 24;
    int64_t n_head_kv = 2;
    int64_t head_dim = 256;
    int64_t idx_q_heads = 4;
    int64_t idx_key_dim = 128;

    // gated residual, on every layer
    int64_t hc = 4;
    int64_t hc_lr = 320;

    // MoE, on every layer
    int64_t n_expert = 512;
    int64_t n_ff = 640;

    constexpr int64_t hc_dim() const { return hc * n_embd; }
    /// `layer % qsa_interval == qsa_interval - 1` is full attention.  Derived, not a second list.
    constexpr int64_t n_qsa_layers() const { return n_layers / qsa_interval; }
    constexpr int64_t n_gdn_layers() const { return n_layers - n_qsa_layers(); }
};

}  // namespace strata::core

// The shape the Qwen3.8 engine's kernels are compiled for: `kCompiledGeometry`, defined by the geometry header chosen
// at configure time (`-DSTRATA_GEOMETRY=<name>` selects include/strata/core/geometries/<name>.hpp,
// `-DSTRATA_GEOMETRY_FILE=<path>` takes any header; cmake/StrataGeometry.cmake writes geometry_selected.hpp into the
// build tree for either).  With neither option the build uses Flash-Next's header and adds no compile flag, so the
// default build is the one it was.  A kernel that fixes a dimension at compile time asserts it against
// `kCompiledGeometry`; geometry_checks.hpp holds the constraints every shape must meet.  The device-dependent sizes
// (tiles, chunks, work-groups) are not here: they stay what the device's report picks at run time.
#if __has_include("strata/core/geometry_selected.hpp")
#include "strata/core/geometry_selected.hpp"
#else
#include "strata/core/geometries/flash_next.hpp"
#endif
#include "strata/core/geometry_checks.hpp"

namespace strata::core {

/// Fills `g` from a qwen4exp model's GGUF shape (artifact/model_spec.hpp).  The engine's kernels are compiled for
/// the compiled shape (kCompiledGeometry), so every value the file writes must equal its value - except `n_expert`,
/// which a pruned variant (the GSQ-RCO Coder) legitimately lowers.  False, naming the first value that differs,
/// otherwise and for any other architecture.
bool geometry_from_spec(const strata::ModelSpec& s, ModelGeometry& g, std::string& err);

/// True for the full-attention layers.  `docs/semantics.md` gives this twice over - `full_attention_interval
/// = 4` and an explicit `attention.compress_ratios` array - and this is the first of the two.
inline bool is_qsa_layer(const ModelGeometry& g, int64_t layer) {
    return layer % g.qsa_interval == g.qsa_interval - 1;
}

/// One layer's tensors, resolved by NAME.  `get("attn_qkv.weight")` looks up `blk.<layer>.attn_qkv.weight`
/// and returns null if the pack does not have it - a null is information, because a GDN layer has no
/// `attn_q` and a QSA layer has no `attn_qkv`.
///
/// Holds a reference to the table; the table must outlive it.
class LayerView {
public:
    LayerView(const WeightTable& table, int64_t layer) : table_(&table), layer_(layer) {}

    int64_t layer() const { return layer_; }
    std::string name(const char* suffix) const;
    const WeightRef* get(const char* suffix) const { return table_->find(name(suffix)); }

private:
    const WeightTable* table_;
    int64_t layer_;
};

/// Every shape the kernels depend on, asserted for ONE layer.  Returns false and fills `err` with the first
/// mismatch, naming the tensor, what it has and what is required.
///
/// This is deliberately a separate function from `LayerView`: a caller that only wants the pointers should
/// not pay for the checks, and a caller that wants the checks should get ALL of them rather than the ones
/// its own call site happens to touch.
bool check_layer(const WeightTable& table, const ModelGeometry& g, int64_t layer, std::string& err);

/// `check_layer` over every layer, plus the cross-layer properties: exactly 12 QSA layers at the right
/// indices, every GDN layer having the GDN set and no QSA tensor, and vice versa.  Returns the first failure.
bool check_all(const WeightTable& table, const ModelGeometry& g, std::string& err);

}  // namespace strata::core
