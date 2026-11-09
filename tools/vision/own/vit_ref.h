// tools/vision/own/vit_ref.h - the float32 CPU reference of the Qwen vision tower (own-stack phase 3).
//
// Plain C++ (threads, no ggml, no vendor library): weights converted to float32 at load, every operation in float32 with double sums where
// a sum is long (norms, softmax).  It is the oracle every backend (CUDA, HIP, later CPU/Metal/SYCL) is checked against, and it is checked
// against mtmd's embeddings first.  It is NOT the product CPU path: a loop-and-block GEMM, no vector intrinsics, ~1.8 GB of weights.
//
// What it computes (geometry-driven, see vit_geometry.h):
//   patches (patch x patch x 3, the two temporal taps summed) -> + bias, in 2x2-merge order -> + the position table resized to the patch
//   grid (bilinear, align corners) -> layers x { x += o(attn(LN1 x)); x += down(gelu(up(LN2 x))) } with a 2D rotary on q and k and global
//   attention -> post LayerNorm -> merger (4 patches -> merger_in -> mm.0 -> gelu -> mm.2) -> one out_dim row per 2x2 block, row-major.
#pragma once

#include "gguf_lite.h"
#include "image_prep.h"
#include "vit_geometry.h"

#include <string>
#include <vector>

namespace strata_vit {

class VitRef {
public:
    // Reads every weight of the tower into float32.  `g` must be supported() (the caller checks).
    bool load(const Gguf& gg, const VitGeometry& g, std::string& err);

    // img: a prepared picture (sides multiples of patch * merge).  rows: nx * ny rows of out_dim floats, row i at grid position (i % nx, i / nx).
    // threads <= 0: all hardware threads.
    bool encode(const PreparedImage& img, int threads, std::vector<float>& rows, int& nx, int& ny, std::string& err) const;

    const VitGeometry& geometry() const { return g_; }

private:
    struct Layer {
        std::vector<float> ln1_w, ln1_b, ln2_w, ln2_b, qkv_w, qkv_b, o_w, o_b, up_w, up_b, down_w, down_b;
    };
    VitGeometry g_;
    // STRATA_VIT_REF_ROUND=1 (a diagnostic): each matrix product's input is rounded to its weight's 16-bit type (BF16 or F16) first, as ggml's CPU
    // and CUDA backends do; it shows how much of a difference to mtmd is that rounding and how much is the maths.  0: no rounding (the reference).
    int act_round_ = 0;               // 0 none, 1 BF16, 2 F16
    std::vector<float> patch_w_;      // [width][3 * patch * patch], the taps summed
    std::vector<float> patch_b_, pos_, post_w_, post_b_, mm0_w_, mm0_b_, mm2_w_, mm2_b_;
    std::vector<Layer> layers_;
};

// small helpers the tests and the GPU checks share
double cosine(const float* a, const float* b, size_t n);

}  // namespace strata_vit
