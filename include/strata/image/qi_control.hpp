// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/qi_control.hpp - Qwen-Image-2.1's ControlNet: alibaba-pai's Qwen-Image-2.1-Fun-Controlnet-Union
// (a VACE-style branch of 16 copies of the DiT's block beside its 32), on the GPU beside QiDit.
//
// The branch, as the model card and VideoX-Fun's QwenImage21ControlTransformer2DModel (Apache-2.0) define it and the
// weights' names and shapes fix it: control_img_in maps each image token's conditioning (QiPipeline::use_control: the
// control latent, the keep mask and the masked base latent) to the DiT's width, in the image's rows of a stream as
// long as the sequence (its text rows 0); before_proj of that is added to the sequence the DiT's first block reads,
// and the 16 control blocks (each a DiT block reading the DiT's modulation, rotary tables and attention) run in turn;
// control block k's after_proj, times control_context_scale (the strength), is added to the sequence after the DiT's
// block 2k (the card's control_layers 0, 2, ..., 30).  Beyond the pipeline, which applies the branch at every step,
// the engine applies it only at the steps whose sigma is in [sigma_lo, sigma_hi] (a start and an end of the schedule
// the server maps to sigmas); at the others the DiT runs as it does without the branch, its prefix cache included.
//
// The weights (BF16) are quantized on the host as they are read, as Mage-Flow's DiT is: W8A8's rotated int8 rows
// where the DiT took that form, else Q8_0; the blocks stream through a LayerStream of their own.
#pragma once

#include "strata/image/qi_dit.hpp"
#include "strata/image/qi_stream.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::platform { class MappedFile; }

namespace strata::image {



class QiControl {
public:
    QiControl();
    ~QiControl();
    QiControl(const QiControl&) = delete;
    QiControl& operator=(const QiControl&) = delete;

    /// Reads the branch's safetensors for `dit` (its width and blocks; open first), quantizing its blocks on the host.
    /// `dry`: the sizes only.  False with err when the file is not the branch or does not fit the DiT.
    bool open(const std::string& path, QiDit& dit, std::string& err, bool dry = false);
    const std::string& path() const { return path_; }

    /// The VRAM the weights outside the blocks take (always there), the blocks' total and the two slots they take at
    /// least; the blocks' budget (those that fit stay, the rest stream).
    uint64_t weight_bytes() const { return weight_bytes_; }
    uint64_t block_bytes() const { return stream_.total(); }
    uint64_t block_min() const { return stream_.min_budget(); }
    void set_block_budget(uint64_t bytes) { stream_.configure(bytes, dit_ != nullptr ? dit_->s_ : nullptr); }
    void drop_blocks() { stream_.drop(); }
    uint64_t blocks_vram() const { return stream_.vram(); }
    uint64_t activations_held() const { return (uint64_t) (c_rows_ + hw_) * (dit_ != nullptr ? dit_->C_ : 0) * 4; }
    void drop_block_homes() { stream_.drop_homes(); }
    /// What a request of `rows` sequence rows and `hw` target tokens adds: the stream, the target's mapped
    /// conditioning (kept through the sampling) and the conditioning's tokens while they are mapped.
    uint64_t activation_bytes(int64_t rows, int64_t hw) const;

    /// The request's conditioning `cond` (device FP32, [129][h][w]: the control latent, the kept mask, the masked
    /// latent), mapped once through control_img_in and before_proj; `strength` the skips' scale, on for the steps of
    /// sigma in [sigma_lo, sigma_hi].
    void set_input(const float* cond, int64_t h, int64_t w, float strength, float sigma_lo, float sigma_hi);
    /// No conditioning: the DiT runs alone (its buffers freed).
    void clear();
    bool set() const { return bp_rows_ != nullptr; }
    bool active(float sigma) const { return set() && sigma >= sigma_lo_ && sigma <= sigma_hi_ && strength_ != 0.f; }

    /// QiDit::forward's hooks: the stream started from the sequence the DiT's first block takes (T rows, the target's
    /// from T - h w on), and after the DiT's block `bi` the control block it is paired with run and its skip added.
    void begin(int64_t T);
    void after_block(int bi, int64_t T);
    void drop_activations();
    /// A new image (the DiT's t above its last): the branch's K means made again.
    void reset_means() { kmean_.reset(); }

private:
    struct Block {
        QiDit::W q, k, v, o, gu, down, after;   // d: the offset within the block's region
        uint64_t nq = 0, nk = 0, after_b = 0;   // the FP32 norms' and after_proj's bias's offsets
    };
    void run_block(int k, int64_t T);

    std::string path_;
    QiDit* dit_ = nullptr;
    bool dry_ = false;
    std::unique_ptr<platform::MappedFile> file_;
    std::vector<std::vector<uint8_t>> host_;   // the blocks' quantized bytes (the stream's spans)
    std::vector<Block> blocks_;
    LayerStream stream_;
    std::vector<int> pair_;                    // the DiT's block -> the control block after it (-1: none)
    int64_t in_k_ = 0, in_kp_ = 0;             // control_img_in's inputs (129) and their padding to 8
    uint16_t* in_w_ = nullptr;                 // control_img_in [C][in_kp] and before_proj [C][C], FP16
    uint16_t* bp_w_ = nullptr;
    float *in_b_ = nullptr, *bp_b_ = nullptr;
    uint64_t weight_bytes_ = 0;
    std::vector<void*> owned_;
    float* bp_rows_ = nullptr;   // before_proj(control_img_in(cond)) [hw][C] (its bias not in it)
    float* c_ = nullptr;         // the stream [T][C]
    int64_t hw_ = 0, c_rows_ = 0;
    I8Means kmean_;              // the int8 attention's K means, a block and context (the DiT resets on a new image)
    float strength_ = 1.f, sigma_lo_ = 0.f, sigma_hi_ = 1.f;
};

}  // namespace strata::image
