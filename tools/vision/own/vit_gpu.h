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
// NVIDIA Pascal, compute capability 6.0 (P100): our tower is the default under auto for F16 and BF16 reader files alike (owner 11-11: our own code is the
// default where its accuracy gates pass, even where it is somewhat slower).  P100, 1,024-token picture: Qwen3.8 BF16 1,068 against mtmd-GPU's 1,073 ms; Qwen3.6
// F16 1,064 against 916 ms (mtmd's F16 products sum in F16 at twice the float32 rate, ours in float32), while ours stays at rule 1 0.999865 against mtmd-CPU
// where mtmd-GPU reads 0.982913 (RESULTS-OWN-3V.md 16.1, PART G).  Other Pascal chips (6.1, 6.2: GTX 10xx, Jetson TX2) are not measured: F16 arithmetic runs
// at a small fraction of the float32 rate there, so the comparison may differ, and mtmd stays there up to the default limit.
// Above the default GPU limit of 1,024 image tokens the speed rule gives way to correctness: mtmd's GPU reader drifts from mtmd's CPU path as pictures
// grow (global cosine 0.983 at 2,116 tokens, 0.81 to 0.83 at 16,384 on the A4000, RTX PRO 5000 and P100), ours stays at 0.9996 to 0.99998 (RESULTS-OWN-3V.md
// 16.7), so a reader started with --max-tokens above 1,024 takes our tower on every CUDA class, however slow (P100 at 16,384 tokens: 196.6 s against 129.0 s).
// max_tokens 0 (not given) counts as the default.
constexpr int kVitGpuDefaultLimit = 1024;
inline bool vit_gpu_slower_nvidia(int cc_major, int cc_minor, bool /*bf16_matrices*/, int max_tokens) {
    return max_tokens <= kVitGpuDefaultLimit && cc_major == 6 && cc_minor != 0;
}
// AMD gfx1151 (Strix Halo, Radeon 8060S) takes our tower under `auto` (owner 11-11: "For Strix - use our reader, even if its a bit slower ... Our priority is to
// get off ggml and llama.cpp").  The ROCm package's rocBLAS has tuned gfx1151 kernels for 16-bit results only (BF16 / F16 -> F32 is fallback-only), so on
// gfx1151 the tower asks for F16 results with float32 accumulation (vit_gpu.cu, VitGpu::load): 0.54-0.91 of mtmd-HIP's time on Strix, rule 1 and 2 pass on
// 18 pictures with both models (RESULTS-OWN-3V.md PART F2/F3; with float32 results it was 1.7 to 3.1).  The fused attention (STRATA_VIT_FA, k_attn_fa_gfx11) is on by
// default there as well (PART F8); STRATA_VIT_FA=0 turns it off.
// `gfx` is the device's arch name (hipDeviceProp_t::gcnArchName, possibly with feature suffixes such as ":xnack-").
inline bool vit_gpu_is_gfx1151(const std::string& gfx) { return gfx.rfind("gfx1151", 0) == 0 && (gfx.size() == 7 || gfx[7] == ':'); }
// AMD on Windows (coordinator 11-11, with the default flip to our tower): no card but gfx1151 has been measured with the Windows HIP package, so `auto` keeps
// mtmd's reader on every other Windows HIP card.  Remove when one RX 9000 / RX 7000 run on Windows is in.  gfx1151 is measured there (Strix) and takes ours.
// Linux HIP (gfx12 measured on the R9700) is not affected.
inline bool vit_gpu_hip_windows_not_measured(bool windows, const std::string& gfx) { return windows && !vit_gpu_is_gfx1151(gfx); }
// Starting the GPU context on a GPU that shares the system RAM (DGX Spark GB10, cudaDevAttrIntegrated): a new CUDA context needs physically contiguous memory
// (blocks of 2 MiB and up); when the kernel has almost none free, cudaSetDevice fails with "out of memory" although tens of GiB are free, and the same call 1 to 5 s
// later succeeds (the kernel compacts memory in between; RESULTS-SPARK.md, 11-11: a probe context failed at 5 of 6 starts, every retry within 5 s succeeded).
// So VitGpu::load tries again, up to kVitCtxRetries times, kVitCtxWaitMs apart, and only for that error, on an integrated GPU, in a Linux CUDA build (an AMD APU under
// HIP and a Mac report "integrated" too; they are not this case).  After the last retry it fails as before and the reader falls back as before.  Pure function so the test needs no GPU.
// 12 (gate 11-11: 4 counted starts needed 5, 5, 5 and 6 retries = 5.4 to 6.8 s, about 1.1 to 1.2 s per retry, the 1 s pause plus the failed attempt; 8 left 2 retries of margin).  Worst case, when the
// context never comes: 12 retries, about 13 to 14.5 s of waiting, then the reader fails as before.
constexpr int kVitCtxRetries = 12;
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

    // True on a GPU class where `--tower auto` keeps mtmd (`--tower own` still runs this tower): NVIDIA Pascal 6.1 / 6.2 up to the default limit (not
    // measured; see vit_gpu_slower_nvidia), Windows HIP cards other than gfx1151 (not measured).  `why` says which.
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
