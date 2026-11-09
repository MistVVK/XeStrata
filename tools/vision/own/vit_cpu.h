// tools/vision/own/vit_cpu.h - the fast CPU tower of the Qwen vision model (own-stack phase 3, the product CPU path; VitRef stays the float32 oracle).
//
// The same maths and the same contract as VitRef (vit_ref.h), geometry-driven, no ggml: the weights stay as stored (BF16 or F16, no float32 copy of the
// matrices), the matrix products are register-tiled float32 kernels with one source per ISA tier (AVX-512, AVX2, portable; cpu_kernels.h), the attention is
// tiled with an online softmax (no score matrix, any picture size), the work runs on a small persistent fork-join team (cpu_team.h).
// 16-bit rounding of the matrix inputs and q, k, v (as ggml's CPU path does): ON by default for BF16 weights, OFF for F16 weights (vit_cpu_round_kind);
// STRATA_VIT_REF_ROUND=0 / 1 overrides (the same switch as VitRef's, where it stays off by default).
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

// The 16-bit rounding the CPU tower applies to its matrix inputs and q, k, v: 0 none, 1 BF16, 2 F16.  weight_kind is the matrices' stored type (1 BF16,
// 2 F16, anything else: none).  Default: on for BF16 weights, off for F16; env (STRATA_VIT_REF_ROUND's value, may be null) "0" / "1" overrides.
int vit_cpu_round_kind(int weight_kind, const char* env);

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
