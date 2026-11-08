// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/upscale/esrgan.hpp - RealESRGAN x4plus (Xintao Wang et al., xinntao/Real-ESRGAN, BSD-3-Clause): the
// RRDBNet super-resolution model on the GPU, an upscaler of the common frame (upscale.hpp: native scale 4, run in
// tiles with context).
//
// The network as basicsr's RRDBNet has it (scale 4): conv_first; num_block RRDBs of three residual dense blocks each
// (five 3 x 3 convolutions, the first four LeakyReLU'd (0.2) and their outputs appended to the block's input channels,
// the fifth's output times 0.2 added to the block's input; the RRDB's output times 0.2 added to its input); conv_body
// added to conv_first's output; two nearest 2x upsamplings each followed by a convolution and LeakyReLU; conv_hr with
// LeakyReLU and conv_last.  The convolutions run through the engine's convolution paths (src/image/qi_conv.hpp: oneDNN
// on Intel GPUs, cuDNN on NVIDIA GPUs in contrib-llvm, else im2col and prefill::Gemm), in FP16 with FP32 sums; the trunk
// and the residuals stay FP32.  The weights are the official RealESRGAN_x4plus.pth's params_ema in a safetensors file
// (tools/image/esrgan.py converts it, tensors unchanged); the fifth convolutions' weights are scaled by 0.2 as they are
// read.  Alpha is not the network's: it comes out nearest-upscaled.
#pragma once

#include "strata/upscale/upscale.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace strata::upscale {

class Esrgan final : public UpscaleModel {
public:
    /// Reads the weights (FP32, BF16 or FP16) from `path` and checks the network's shape; std::runtime_error when the
    /// file is not RRDBNet's x4 weights.
    explicit Esrgan(const std::string& path);
    ~Esrgan() override;
    Esrgan(const Esrgan&) = delete;
    Esrgan& operator=(const Esrgan&) = delete;

    const ModelInfo& info() const override { return info_; }
    uint64_t weight_bytes() const override;
    uint64_t clip_bytes(int64_t w, int64_t h, int64_t frames) const override;
    uint64_t clip_largest(int64_t w, int64_t h, int64_t frames) const override;
    void run_clip(const uint8_t* in, int64_t w, int64_t h, int64_t frames, uint8_t* out, uint64_t seed) override;
    void release(bool all) override;
    uint64_t held() const override;

    /// Pixels of the frame read around a tile's core (each side, where the frame has them): the tiles overlap by 32,
    /// each seeing as far past the seam.
    static constexpr int64_t kContext = 32;

private:
    struct Impl;
    ModelInfo info_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::upscale
