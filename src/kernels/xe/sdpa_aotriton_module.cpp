// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/sdpa_aotriton_module.cpp - see include/strata/kernels/sdpa_aotriton.hpp: aotriton's flash attention on AMD
// GPUs (MIT; its library and its kernels' images beside the engine), on the queue's HIP stream (while the queue records
// a SYCL graph, in a native command whose stream the graph takes in).  A module of its own
// (libxestrata_sdpa_aotriton.so, xestrata_sdpa_aotriton.dll), which the kernels open on an AMD GPU only
// (kernels::aotriton_sdpa): aotriton needs AMD's HIP runtime, which a machine with other makers' GPUs need not have,
// and the engine must start there.

#include <aotriton/flash.h>
#include <hip/hip_runtime.h>
#include <sycl/ext/oneapi/backend/hip.hpp>


#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace strata::image::sdpa {

namespace {

std::mutex g_mu;

struct Aot {
    bool failed = false;
    int32_t* counter = nullptr;   // the persistent kernels' work counter, zeroed before each call
};
std::unordered_map<sycl::device, Aot> g_aot;

}  // namespace

// `rows` query rows over `keys` keys, the keys' kv_heads heads each read by heads / kv_heads of the queries' (GQA);
// causal: row t sees the keys up to keys - rows + t, with a window the last `window` of them (kernels::aotriton_sdpa)
bool aotriton(sycl::queue& q, const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t rows, int64_t keys,
              int64_t heads, int64_t kv_heads, int64_t hd, uint16_t* out16, bool causal, int64_t window) {
    namespace ao = aotriton;
    std::lock_guard<std::mutex> lock(g_mu);
    Aot& d = g_aot[q.get_device()];
    if (d.failed) return false;
    if (d.counter == nullptr) d.counter = sycl::malloc_device<int32_t>(1, q);
    auto view = [&](const uint16_t* p, int64_t n, int64_t h) {
        const uint64_t row = (uint64_t) (h * hd);
        return ao::TensorView<4>(reinterpret_cast<intptr_t>(p),
                                 std::array<uint64_t, 4>{1, (uint64_t) h, (uint64_t) n, (uint64_t) hd},
                                 std::array<uint64_t, 4>{(uint64_t) n * row, (uint64_t) hd, row, 1}, ao::kFloat16);
    };
    ao::v3::flash::attn_fwd_params p;
    p.Q = view(q16, rows, heads);
    p.K = view(k16, keys, kv_heads);
    p.V = view(v16, keys, kv_heads);
    p.Out = view(out16, rows, heads);
    p.B = ao::TensorView<4>::get_null_tensor(ao::kFloat16);
    p.A = ao::TensorView<2>::get_null_tensor(ao::kFloat16);
    p.L = ao::TensorView<2>::get_null_tensor(ao::kFloat32);
    p.Sm_scale = 1.0f / std::sqrt((float) hd);
    p.dropout_p = 0.0f;
    p.philox_seed_ptr = ao::TensorView<0>::get_null_tensor(ao::kUInt64);
    p.philox_offset1 = ao::TensorView<0>::get_null_tensor(ao::kUInt64);
    p.philox_offset2 = 0;
    p.philox_seed_output = ao::TensorView<0>::get_null_tensor(ao::kUInt64);
    p.philox_offset_output = ao::TensorView<0>::get_null_tensor(ao::kUInt64);
    p.encoded_softmax = ao::TensorView<4>::get_null_tensor(ao::kFloat16);
    p.persistent_atomic_counter = ao::TensorView<0>(reinterpret_cast<intptr_t>(d.counter), ao::kInt32);
    // causal: the windowed form (the binary kernels have no other causal), its edges counted from the top-left
    // diagonal (row t sees the keys t - window_left .. t + window_right): both that diagonal where the rows are the keys,
    // else a window of every key before and keys - rows after, the bottom-right diagonal; with a sliding window
    // window - 1 - (keys - rows) before (the caller keeps keys - rows below the window)
    p.causal_type = causal ? ao::v3::flash::CausalType::WindowedAttention : ao::v3::flash::CausalType::None;
    if (causal && window > 0 && keys - rows >= window) return false;
    const bool square = rows == keys && window <= 0;
    p.window_left = !causal ? 0
                    : square ? ao::v3::flash::WindowValue::TopLeftAligned
                    : window > 0 ? (int32_t) (window - 1 - (keys - rows))
                                 : (int32_t) keys;
    p.window_right = !causal ? 0 : square ? ao::v3::flash::WindowValue::TopLeftAligned : (int32_t) (keys - rows);
    p.Max_seqlen_q = (int32_t) rows;
    p.Max_seqlen_k = (int32_t) keys;
    q.memset(d.counter, 0, sizeof(int32_t));
    namespace sx = sycl::ext::oneapi::experimental;
    if (q.ext_oneapi_get_state() == sx::queue_state::recording) {
        // called on the queue's stream straight, the kernel would run once, at the recording, and not at the replays:
        // a native command whose stream is taken into the graph's HIP graph, as oneMath's backends do (HIP 6.2)
#if HIP_VERSION >= 60200000
        q.submit([&](sycl::handler& h) {
            h.ext_codeplay_enqueue_native_command([p](const sycl::interop_handle& ih) {
                const hipStream_t hs = ih.get_native_queue<sycl::backend::ext_oneapi_hip>();
                hipGraph_t g = nullptr;
                if (ih.ext_codeplay_has_graph()) g = ih.ext_codeplay_get_native_graph<sycl::backend::ext_oneapi_hip>();
                if (g != nullptr &&
                    hipStreamBeginCaptureToGraph(hs, g, nullptr, nullptr, 0, hipStreamCaptureModeGlobal) != hipSuccess)
                    throw std::runtime_error("aotriton: hipStreamBeginCaptureToGraph failed");
                const hipError_t e = ao::v3::flash::attn_fwd(p, ao::v3::flash::attn_fwd_params::kVersion,
                                                             ao::Stream(hs));
                if (g != nullptr && hipStreamEndCapture(hs, &g) != hipSuccess)
                    throw std::runtime_error("aotriton: hipStreamEndCapture failed");
                if (e != hipSuccess)
                    throw std::runtime_error("aotriton's attention: HIP error " + std::to_string((int) e));
            });
        });
        return true;
#else
        return false;   // the engine's own kernel, which the graph records
#endif
    }
    const hipStream_t hs = sycl::get_native<sycl::backend::ext_oneapi_hip>(q);
    const hipError_t e = ao::v3::flash::attn_fwd(p, ao::v3::flash::attn_fwd_params::kVersion, ao::Stream(hs));
    if (e != hipSuccess) {
        std::fprintf(stderr, "xestrata-image: aotriton's attention is not taken on this GPU (HIP error %d): the engine's "
                             "own kernel\n", (int) e);
        d.failed = true;
        return false;
    }
    return true;
}


// A training step's attention (no mask, kv_heads = heads): the forward with each row's log-sum-exp of its scaled
// scores, lse [heads][rows] FP32 (aotriton's own layout), and the backward from the forward's output, its gradient do16
// (FP16, in any one unit: the gradients come out in it) and lse into dq16 [rows][heads * hd], dk16 and dv16 [keys][...]
// FP16; ws: the backward's own D [heads][rows] and dQ's FP32 sums [rows][heads * hd] (zeroed here)
bool aotriton_train(sycl::queue& q, bool bwd, const uint16_t* q16, const uint16_t* k16, const uint16_t* v16,
                    const uint16_t* o16, const uint16_t* do16, float* lse, int64_t rows, int64_t keys, int64_t heads,
                    int64_t hd, uint16_t* out16, uint16_t* dq16, uint16_t* dk16, uint16_t* dv16, float* ws) {
    namespace ao = aotriton;
    namespace sx = sycl::ext::oneapi::experimental;
    if (q.ext_oneapi_get_state() == sx::queue_state::recording) return false;   // a training step records no graph
    std::lock_guard<std::mutex> lock(g_mu);
    Aot& d = g_aot[q.get_device()];
    if (d.failed) return false;
    if (d.counter == nullptr) d.counter = sycl::malloc_device<int32_t>(1, q);
    const uint64_t row = (uint64_t) (heads * hd);
    auto view = [&](const void* p, int64_t n, ao::DType t) {
        return ao::TensorView<4>(reinterpret_cast<intptr_t>(p),
                                 std::array<uint64_t, 4>{1, (uint64_t) heads, (uint64_t) n, (uint64_t) hd},
                                 std::array<uint64_t, 4>{(uint64_t) n * row, (uint64_t) hd, row, 1}, t);
    };
    auto rows2 = [&](const float* p) {
        return ao::TensorView<2>(reinterpret_cast<intptr_t>(p), std::array<uint64_t, 2>{(uint64_t) heads, (uint64_t) rows},
                                 std::array<uint64_t, 2>{(uint64_t) rows, 1}, ao::kFloat32);
    };
    const hipStream_t hs = sycl::get_native<sycl::backend::ext_oneapi_hip>(q);
    hipError_t e = hipSuccess;
    if (!bwd) {
        ao::v3::flash::attn_fwd_params p;
        p.Q = view(q16, rows, ao::kFloat16);
        p.K = view(k16, keys, ao::kFloat16);
        p.V = view(v16, keys, ao::kFloat16);
        p.Out = view(out16, rows, ao::kFloat16);
        p.B = ao::TensorView<4>::get_null_tensor(ao::kFloat16);
        p.A = ao::TensorView<2>::get_null_tensor(ao::kFloat16);
        p.L = rows2(lse);
        p.Sm_scale = 1.0f / std::sqrt((float) hd);
        p.dropout_p = 0.0f;
        p.philox_seed_ptr = ao::TensorView<0>::get_null_tensor(ao::kUInt64);
        p.philox_offset1 = ao::TensorView<0>::get_null_tensor(ao::kUInt64);
        p.philox_offset2 = 0;
        p.philox_seed_output = ao::TensorView<0>::get_null_tensor(ao::kUInt64);
        p.philox_offset_output = ao::TensorView<0>::get_null_tensor(ao::kUInt64);
        p.encoded_softmax = ao::TensorView<4>::get_null_tensor(ao::kFloat16);
        p.persistent_atomic_counter = ao::TensorView<0>(reinterpret_cast<intptr_t>(d.counter), ao::kInt32);
        p.causal_type = ao::v3::flash::CausalType::None;
        p.window_left = p.window_right = 0;
        p.Max_seqlen_q = (int32_t) rows;
        p.Max_seqlen_k = (int32_t) keys;
        q.memset(d.counter, 0, sizeof(int32_t));
        e = ao::v3::flash::attn_fwd(p, ao::v3::flash::attn_fwd_params::kVersion, ao::Stream(hs));
    } else {
        float* dsum = ws;                       // D [heads][rows]
        float* dqacc = ws + heads * rows;       // dQ's FP32 sums
        q.memset(dqacc, 0, (size_t) (rows * heads * hd) * 4);
        ao::v3::flash::attn_bwd_params p;
        p.Q = view(q16, rows, ao::kFloat16);
        p.K = view(k16, keys, ao::kFloat16);
        p.V = view(v16, keys, ao::kFloat16);
        p.B = ao::TensorView<4>::get_null_tensor(ao::kFloat16);
        p.Sm_scale = 1.0f / std::sqrt((float) hd);
        p.Out = view(o16, rows, ao::kFloat16);
        p.DO = view(do16, rows, ao::kFloat16);
        p.DK = view(dk16, keys, ao::kFloat16);
        p.DV = view(dv16, keys, ao::kFloat16);
        p.DQ = view(dq16, rows, ao::kFloat16);
        p.DB = ao::TensorView<4>::get_null_tensor(ao::kFloat16);
        p.L = rows2(lse);
        p.D.eager = rows2(dsum);
        p.DQ_ACC.eager = view(dqacc, rows, ao::kFloat32);
        p.dropout_p = 0.0f;
        p.philox_seed_ptr = ao::TensorView<0>::get_null_tensor(ao::kUInt64);
        p.philox_offset1 = ao::TensorView<0>::get_null_tensor(ao::kUInt64);
        p.philox_offset2 = 0;
        p.causal_type = ao::v3::flash::CausalType::None;
        p.window_left = p.window_right = 0;
        p.Max_seqlen_q = (int32_t) rows;
        p.Max_seqlen_k = (int32_t) keys;
        e = ao::v3::flash::attn_bwd(p, ao::v3::flash::attn_bwd_params::kVersion, ao::Stream(hs));
    }
    if (e != hipSuccess) {
        std::fprintf(stderr, "xestrata-image: aotriton's training attention is not taken on this GPU (HIP error %d)\n",
                     (int) e);
        return false;
    }
    return true;
}

}  // namespace strata::image::sdpa

// the module's entry, which qi_sdpa.cpp looks up by this name: q a sycl::queue of the same SYCL runtime
#if defined(_WIN32)
#define STRATA_SDPA_EXPORT extern "C" __declspec(dllexport)
#else
#define STRATA_SDPA_EXPORT extern "C" __attribute__((visibility("default")))
#endif
STRATA_SDPA_EXPORT bool strata_sdpa_aotriton(void* q, const uint16_t* q16, const uint16_t* k16, const uint16_t* v16,
                                             int64_t rows, int64_t keys, int64_t heads, int64_t kv_heads, int64_t hd,
                                             uint16_t* out16, bool causal, int64_t window) {
    return strata::image::sdpa::aotriton(*static_cast<sycl::queue*>(q), q16, k16, v16, rows, keys, heads, kv_heads, hd,
                                         out16, causal, window);
}
STRATA_SDPA_EXPORT bool xestrata_sdpa_aotriton_train(void* q, bool bwd, const uint16_t* q16, const uint16_t* k16,
                                                   const uint16_t* v16, const uint16_t* o16, const uint16_t* do16,
                                                   float* lse, int64_t rows, int64_t keys, int64_t heads, int64_t hd,
                                                   uint16_t* out16, uint16_t* dq16, uint16_t* dk16, uint16_t* dv16,
                                                   float* ws) {
    return strata::image::sdpa::aotriton_train(*static_cast<sycl::queue*>(q), bwd, q16, k16, v16, o16, do16, lse, rows,
                                               keys, heads, hd, out16, dq16, dk16, dv16, ws);
}
