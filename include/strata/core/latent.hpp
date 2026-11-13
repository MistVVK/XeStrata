// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 Edge0
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT AND Apache-2.0
// include/strata/core/latent.hpp - the latent MoE pack's readers (Edge0-145B / Edge0-120B, Qwen3.8-Flash-Next's
// shell with experts at a latent width): the D/U dense file, the experts' A file, the group labels, the
// native_experts.txt v6 manifest and the Q3_K column scales.  Written after edge0-strata's latent_dense.cpp,
// its labels reads (generate.cpp / verify.cpp / prefill.cpp) and expert_layout.cpp's v6 / scales parts.
//
// The layer's computation: x [H] -> D[group] (ELL x H) -> the router's experts, each ELL -> I -> ELL on its
// group's latent row -> U[group] (H x ELL) -> the weighted sum.  D and U are one pair a layer (G groups);
// only the experts work at ELL.  Every reader refuses a file whose size disagrees with its layout: a misread
// layout would produce garbage answers, so there is no silent fallback.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::platform {
class MappedFile;
}

namespace strata::core {

/// The D/U dense file: layer l is D [G*ELL, H] then U [G*H, ELL], rows of Q8_0 (fmt 8, 34 B a 32-value block)
/// or Q4_0 (fmt 4, 18 B).  The format comes from XESTRATA_LATENT_DENSE_FMT (STRATA_LATENT_DENSE_FMT) at open,
/// default 8, and the file's size must match it.  Mapped read-only; the engine uploads a layer's pair on demand.
class LatentDense {
public:
    LatentDense();
    ~LatentDense();
    LatentDense(const LatentDense&) = delete;
    LatentDense& operator=(const LatentDense&) = delete;

    bool open(const std::string& path, int64_t n_layers, int64_t n_groups, int64_t n_embd, int64_t latent_dim,
              std::string& err);
    void close();

    /// The file's size for a geometry and format (what open checks against).
    static uint64_t bytes_for(int64_t n_layers, int64_t n_groups, int64_t n_embd, int64_t latent_dim, int fmt);

    bool mapped() const { return map_ != nullptr; }
    int64_t layers() const { return n_layers_; }
    int64_t groups() const { return n_groups_; }
    int64_t embd() const { return n_embd_; }
    int64_t latent() const { return latent_dim_; }
    int fmt() const { return fmt_; }
    size_t row_b() const { return fmt_ == 4 ? 18 : 34; }
    size_t d_row_bytes() const { return (size_t) (n_embd_ / 32) * row_b(); }
    size_t u_row_bytes() const { return (size_t) (latent_dim_ / 32) * row_b(); }
    size_t d_bytes() const { return (size_t) n_groups_ * (size_t) latent_dim_ * d_row_bytes(); }
    size_t u_bytes() const { return (size_t) n_groups_ * (size_t) n_embd_ * u_row_bytes(); }
    uint64_t bytes() const { return map_ ? map_size_ : 0; }

    const uint8_t* d(int64_t layer) const;
    const uint8_t* u(int64_t layer) const;

private:
    std::unique_ptr<platform::MappedFile> map_;
    uint64_t map_size_ = 0;
    int64_t n_layers_ = 0, n_groups_ = 0, n_embd_ = 0, latent_dim_ = 0;
    int fmt_ = 8;
};

/// The experts' A file (experts.bin): layer-major, expert-major, one fixed-size blob an expert
/// (native_expert_layout(gu_type, d_type, ELL, I): gate rows, up rows, down rows, ggml blocks as they are).
class LatentA {
public:
    LatentA();
    ~LatentA();
    LatentA(const LatentA&) = delete;
    LatentA& operator=(const LatentA&) = delete;

    bool open(const std::string& path, int64_t n_layers, int64_t n_expert, int64_t blob_bytes, std::string& err);
    void close();
    bool mapped() const { return map_ != nullptr; }
    const uint8_t* blob(int64_t layer, int64_t expert) const;
    int64_t blob_bytes() const { return blob_bytes_; }

private:
    std::unique_ptr<platform::MappedFile> map_;
    int64_t n_layers_ = 0, n_expert_ = 0, blob_bytes_ = 0;
};

/// One layer of native_experts.txt: the gate/up and down ggml types, the layer's offset in the A file and one
/// expert's blob bytes.
struct LatentLayerFmt {
    int gu_type = 0, d_type = 0;
    uint64_t offset = 0, blob_bytes = 0;
};

/// native_experts.txt, versions 3 to 6 (the line format they share: layer gu_type d_type offset blob_bytes,
/// then v3/v4's optional role offsets and shard).  v6 is the latent A-only pack: its header names expert_embd
/// and expert_ff (the experts' width, the shell's hidden width when absent) and it carries no hadamard.txt.
/// XeStrata's own xs1/xs2 manifests stay with expert_layout_load; this reader is the latent pack's.
struct LatentManifest {
    int version = 0;
    int64_t expert_embd = 0, expert_ff = 0;   // 0 when the header does not name them
    int64_t n_expert = 0;                     // the header's (n_expert N), 0 when absent
    std::vector<LatentLayerFmt> layers;
};
bool latent_manifest_read(const std::string& path, LatentManifest& m, std::string& err);

/// labels.bin: one int32 group label per (layer, expert), rows of n_expert.  The file must be exactly
/// n_layers * n_expert * 4 bytes and every label in [0, n_groups).
bool latent_labels_read(const std::string& path, int64_t n_layers, int64_t n_expert, int n_groups,
                        std::vector<int32_t>& labels, std::string& err);

/// The Q3_K shared column scales (scales/l{l}.f32 beside the manifest): the gate/up rows are A*diag(s), so
/// every consumer of the D stage's rows multiplies by 1/s first.  All layers or none: a half-scaled pack is
/// refused.  `inv` comes back n_layers * width floats (layer-major), empty when the pack has no scales.
/// A scale that is not positive reads as 1 (edge0-strata's rule).
bool latent_colscale_read(const std::string& pack_dir, int64_t n_layers, int64_t width, std::vector<float>& inv,
                          std::string& err);

}  // namespace strata::core