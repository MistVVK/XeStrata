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
// NVIDIA Pascal (compute capability 6.x): no tensor cores; mtmd's products of F16 matrices accumulate in F16 at twice the float32 rate, ours in float32 (an
// F16-accumulating variant of ours was as fast but missed rule 1: 0.9972), so with F16 matrices (Qwen3.6) ours stays slower (P100: at best 1.10 of mtmd-GPU,
// RESULTS-OWN-3V.md 16.1).  With BF16 matrices (Qwen3.8) mtmd multiplies in float32 too, and ours (BF16 held as F16, the larger score chunk) passes the whole
// gate on a P100 (cc 6.0: 0.31 to 0.99 of mtmd-GPU's time, 16.1).  Other Pascal chips (6.1, 6.2: GTX 10xx, Jetson TX2) are not measured: mtmd stays there.
// Above the default GPU limit of 1,024 image tokens the speed rule gives way to correctness: mtmd's GPU reader drifts from mtmd's CPU path as pictures
// grow (global cosine 0.983 at 2,116 tokens, 0.81 to 0.83 at 16,384 on the A4000, RTX PRO 5000 and P100), ours stays at 0.9996 to 0.99998 (RESULTS-OWN-3V.md
// 16.7), so a reader started with --max-tokens above 1,024 takes our tower on every CUDA class, however slow (P100 at 16,384 tokens: 196.6 s against 129.0 s).
// max_tokens 0 (not given) counts as the default.  AMD gfx1151 keeps its rule (no large-picture data there yet).
constexpr int kVitGpuDefaultLimit = 1024;
inline bool vit_gpu_slower_nvidia(int cc_major, int cc_minor, bool bf16_matrices, int max_tokens) {
    return max_tokens <= kVitGpuDefaultLimit && cc_major == 6 && !(cc_minor == 0 && bf16_matrices);
}
// AMD gfx1151 (Strix Halo, Radeon 8060S): the ROCm package's rocBLAS ships only its generic "fallback" Tensile kernels for gfx1151, so every hipBLAS product runs
// untuned (Windows, Strix: 1.73 to 3.03 of mtmd-HIP's time; the weight products alone 730 ms of a 1,024-token picture against mtmd-HIP's 690 ms for all of it;
// RESULTS-OWN-3V.md 12.14).  Windows and Linux alike until a Linux gfx1151 box is measured.  `gfx` is the device's arch name (hipDeviceProp_t::gcnArchName,
// possibly with feature suffixes such as ":xnack-").
inline bool vit_gpu_slower_amd(const std::string& gfx) { return gfx.rfind("gfx1151", 0) == 0 && (gfx.size() == 7 || gfx[7] == ':'); }
// AMD on Windows, every card (coordinator 11-11, with the default flip to our tower): the Windows HIP package's rocBLAS is the one whose gfx1151 kernels are
// fallback-only, and no other card has been measured with that package, so `auto` keeps mtmd's reader on Windows HIP.  Not measured on Windows HIP; remove when
// one RX 9000 / RX 7000 run on Windows is in.  Linux HIP (gfx12 measured on the R9700) is not affected.
inline bool vit_gpu_hip_windows_not_measured(bool windows) { return windows; }
// Starting the GPU context on a GPU that shares the system RAM (DGX Spark GB10, cudaDevAttrIntegrated): a new CUDA context needs physically contiguous memory
// (blocks of 2 MiB and up); when the kernel has almost none free, cudaSetDevice fails with "out of memory" although tens of GiB are free, and the same call 1 to 5 s
// later succeeds (the kernel compacts memory in between; RESULTS-SPARK.md, 11-11: a probe context failed at 5 of 6 starts, every retry within 5 s succeeded).
// So VitGpu::load tries again, up to kVitCtxRetries times, kVitCtxWaitMs apart, and only for that error, on an integrated GPU, in a Linux CUDA build (an AMD APU under
// HIP and a Mac report "integrated" too; they are not this case).  After the last retry it fails as before and the reader falls back as before.  Pure function so the test needs no GPU.
constexpr int kVitCtxRetries = 8;
constexpr int kVitCtxWaitMs = 1000;
inline bool vit_gpu_retry_context(bool linux_cuda_build, bool integrated_gpu, bool out_of_memory, int retries_done) {
    return linux_cuda_build && integrated_gpu && out_of_memory && retries_done >= 0 && retries_done < kVitCtxRetries;
}

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
    static bool prefer_mtmd(bool bf16_matrices, int max_tokens, std::string& why);      // bf16_matrices: the mmproj's layer matrices are BF16; max_tokens: --max-tokens (0: not given)

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
