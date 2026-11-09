// tools/vision/own/vit_cpu.h - the fast CPU tower of the Qwen vision model (own-stack phase 3, the product CPU path; VitRef stays the float32 oracle).
//
// The same maths and the same contract as VitRef (vit_ref.h), geometry-driven, no ggml: the weights stay as stored (BF16 or F16, no float32 copy of the
// matrices), the matrix products are register-tiled float32 kernels with one source per ISA tier (AVX-512, AVX2, portable; cpu_kernels.h), the attention is
// tiled with an online softmax (no score matrix, any picture size), the work runs on a small persistent fork-join team (cpu_team.h).
// STRATA_VIT_REF_ROUND=1 (off by default) rounds the matrix inputs and q, k, v to 16 bits as ggml's CPU path does (same switch as VitRef's).
#pragma once

#include "cpu_kernels.h"
#include "cpu_team.h"
#include "gguf_lite.h"
#include "image_prep.h"
#include "vit_geometry.h"

#include <memory>
#include <string>
#include <vector>

namespace strata_vit {

class VitCpu {
public:
    VitCpu();
    ~VitCpu();
    // Reads the tower's weights (16-bit matrices as stored).  `g` must be supported() (the caller checks).
    bool load(const Gguf& gg, const VitGeometry& g, std::string& err);

    // img: a prepared picture.  rows: nx * ny rows of out_dim floats.  threads <= 0: half the hardware threads, at most 16.
    bool encode(const PreparedImage& img, int threads, std::vector<float>& rows, int& nx, int& ny, std::string& err);

    const VitGeometry& geometry() const;
    const char* isa_name() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata_vit
