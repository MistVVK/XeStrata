// tools/vision/own/vit_gpu.h - the Qwen vision tower on one GPU, CUDA and HIP from one source (own-stack phase 3).
//
// The same maths as VitRef (vit_ref.h), the same geometry (vit_geometry.h): weights stay on the device as BF16 or F16 (as the file stores
// them; F32 matrices become BF16, Q8_0 becomes F16), the activations are float32, each matrix product is a cuBLAS / hipBLAS GemmEx with a
// 16-bit input cast from the float32 activations and float32 accumulation (what ggml's CUDA backend does for the same weights), and the
// attention is a tiled online-softmax kernel in float32 (head size 72 does not fit the vendor flash kernels, and a materialised score
// matrix would cost 1 GB at 4,096 tokens).  Built only with STRATA_VIT_CUDA or STRATA_VIT_HIP; without either the functions say so.
#pragma once

#include "gguf_lite.h"
#include "image_prep.h"
#include "vit_geometry.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace strata_vit {

struct VitGpuImpl;

class VitGpu {
public:
    VitGpu();
    ~VitGpu();
    VitGpu(const VitGpu&) = delete;
    VitGpu& operator=(const VitGpu&) = delete;

    // True when this build has a GPU backend and a device to run it on; `why` says what is missing.
    static bool available(std::string& why);

    // Uploads the weights to `device` and sizes the work buffers for pictures of up to `max_patches` patches (grown by encode when a picture
    // needs more).  `g` must be supported().
    bool load(const Gguf& gg, const VitGeometry& g, int device, int max_patches, std::string& err);

    // Same contract as VitRef::encode.
    bool encode(const PreparedImage& img, std::vector<float>& rows, int& nx, int& ny, std::string& err);

    // Bytes of device memory held now (weights + work buffers).
    size_t device_bytes() const;

private:
    std::unique_ptr<VitGpuImpl> impl_;
};

}  // namespace strata_vit
