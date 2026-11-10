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

// The GPU classes where this tower is known to be slower than mtmd's GPU reader (VitGpu::prefer_mtmd uses these; pure functions, so the tests need no GPU).
// NVIDIA Pascal (compute capability 6.x): no tensor cores; mtmd's F16 products accumulate in F16 at twice the float32 rate, ours in float32 (P100: 1.15 to 1.38 of
// mtmd-GPU's time on pictures of 260 tokens or more, RESULTS-OWN-3V.md 12.1).
inline bool vit_gpu_slower_nvidia(int cc_major) { return cc_major == 6; }
// AMD gfx1151 (Strix Halo, Radeon 8060S): the ROCm package's rocBLAS ships only its generic "fallback" Tensile kernels for gfx1151, so every hipBLAS product runs
// untuned (Windows, Strix: 1.73 to 3.03 of mtmd-HIP's time; the weight products alone 730 ms of a 1,024-token picture against mtmd-HIP's 690 ms for all of it;
// RESULTS-OWN-3V.md 12.14).  Windows and Linux alike until a Linux gfx1151 box is measured.  `gfx` is the device's arch name (hipDeviceProp_t::gcnArchName,
// possibly with feature suffixes such as ":xnack-").
inline bool vit_gpu_slower_amd(const std::string& gfx) { return gfx.rfind("gfx1151", 0) == 0 && (gfx.size() == 7 || gfx[7] == ':'); }

class VitGpu {
public:
    VitGpu();
    ~VitGpu();
    VitGpu(const VitGpu&) = delete;
    VitGpu& operator=(const VitGpu&) = delete;

    // True when this build has a GPU backend and a device to run it on; `why` says what is missing.
    static bool available(std::string& why);

    // True on a GPU class where this tower is known to be slower than mtmd's GPU reader, so `--tower auto` keeps mtmd there (`--tower own` still runs
    // this tower): NVIDIA Pascal (compute capability 6.x: no tensor cores; mtmd's F16 products accumulate in F16 at twice the float32 rate, ours in float32;
    // measured on a P100, RESULTS-OWN-3V.md 12.1).  `why` says which.
    static bool prefer_mtmd(std::string& why);

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
