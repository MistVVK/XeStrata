// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_sdpa_aotriton.cpp - see include/strata/kernels/sdpa_aotriton.hpp: aotriton's flash attention on AMD
// GPUs (MIT; its library and its kernels' images beside the engine), on the queue's HIP stream (while the queue records
// a SYCL graph, in a native command whose stream the graph takes in).  A module of its own
// (libstrata_sdpa_aotriton.so, strata_sdpa_aotriton.dll), which the kernels open on an AMD GPU only
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
// causal: row t sees the keys up to keys - rows + t
bool aotriton(sycl::queue& q, const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t rows, int64_t keys,
              int64_t heads, int64_t kv_heads, int64_t hd, uint16_t* out16, bool causal) {
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
    // causal: the windowed form (the binary kernels have no other causal), both edges the top-left aligned diagonal
    // where the rows are the keys, else a window of every key before and keys - rows after, the bottom-right diagonal
    p.causal_type = causal ? ao::v3::flash::CausalType::WindowedAttention : ao::v3::flash::CausalType::None;
    const bool square = rows == keys;
    p.window_left = !causal ? 0 : square ? ao::v3::flash::WindowValue::TopLeftAligned : (int32_t) keys;
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
        std::fprintf(stderr, "strata-image: aotriton's attention is not taken on this GPU (HIP error %d): the engine's "
                             "own kernel\n", (int) e);
        d.failed = true;
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
                                             uint16_t* out16, bool causal) {
    return strata::image::sdpa::aotriton(*static_cast<sycl::queue*>(q), q16, k16, v16, rows, keys, heads, kv_heads, hd,
                                         out16, causal);
}
