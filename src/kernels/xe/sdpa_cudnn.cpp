// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/sdpa_cudnn.cpp - see sdpa_cudnn_internal.hpp, qi_conv.hpp and strata/kernels/sdpa_cudnn.hpp: cuDNN's SDPA and
// convolution through cudnn-frontend on NVIDIA GPUs (contrib-llvm), built into the kernels' library (the plain models'
// prompt shares the SDPA, as oneDNN's), cuDNN opened at the first use (the user's installation, not free software:
// open_cudnn), on the queue's CUDA stream through a native command (a SYCL graph being recorded takes it), a cuDNN
// graph built once a shape and kept.

#include <cudnn_frontend.h>   // with its dynamic loading on Linux
#include <sycl/backend/cuda.hpp>
#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>

#include <filesystem>
// cuDNN and the CUDA runtime are linked and loaded at their first call (CMakeLists.txt: cudnn-frontend's
// own loading takes only CUDA 13's runtime on Windows), after open_cudnn has loaded them from where the user put them
namespace cudnn_frontend {
HMODULE cudnn_dlhandle = nullptr;   // cuDNN opened at the first attention (this file's, as on Linux)
}
#else
#include <dlfcn.h>
namespace cudnn_frontend {
void* cudnn_dlhandle = nullptr;
}
#endif

#include "../../image/qi_conv.hpp"
#include "strata/core/graph.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/sdpa_cudnn.hpp"
#include "strata/platform/downgrade.hpp"
#include "strata/platform/env.hpp"
#include "strata/kernels/sdpa_cudnn_internal.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace strata::image::sdpa {

namespace {

std::mutex g_mu;
// the handle's stream set and the work put on it under this lock: the native commands of two queues run on the
// runtime's host threads at once, and the handle is one a device - without it one command's execute could go on the
// other's stream (the GPU work itself runs on, the workspaces apart below)
std::mutex g_exec_mu;

namespace fe = cudnn_frontend;

struct Cudnn {
    bool failed = false, conv_failed = false;   // conv_failed: cuDNN not opened for the convolutions
    cudnnHandle_t handle = nullptr;
    // (rows, keys, heads, kv heads, hd, batch, causal, window) -> its graph
    using Key = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t>;
    std::map<Key, std::shared_ptr<fe::graph::Graph>> graphs;
    std::list<Key> graph_lru;
    // a training step's: (rows, keys, heads, hd, backward) -> its graph
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, bool>, std::shared_ptr<fe::graph::Graph>> train;
    std::map<std::vector<int64_t>, std::shared_ptr<fe::graph::Graph>> convs;   // null: no plan for the shape
};
std::unordered_map<sycl::device, Cudnn> g_cudnn;

// The workspace of a queue's cuDNN work: the queue's own belonging (Runtime::queue_item), freed when its stream is
// destroyed.  One a device, two streams on the GPU ran their graphs on it at once, and a reallocation waited for the
// calling queue alone while the other's work still read it; a queue's own, the wait covers every reader and the
// streams never share it (a recorded graph's replay holds its queue's workspace, the epoch bump dropping the
// recording when it is taken anew).
struct CudnnWs {
    explicit CudnnWs(sycl::queue& q) : q(q) {}
    ~CudnnWs() {
        if (ws != nullptr) {
            sycl::free(ws, q);
            core::graph_epoch_bump();   // a recorded step may hold the old one
        }
    }
    sycl::queue q;
    void* ws = nullptr;
    int64_t ws_bytes = 0;
};

constexpr int64_t kQ = 1, kK = 2, kV = 3, kO = 4;   // the tensors' ids in the variant pack
constexpr int64_t kStats = 5, kDO = 6, kDQ = 7, kDK = 8, kDV = 9;   // a training step's

// The driver's stream capture into a given CUDA graph (CUDA 12.3), from the driver the SYCL runtime has loaded
struct Capture {
#if CUDA_VERSION >= 12030
    CUresult (*begin)(CUstream, CUgraph, const CUgraphNode*, const CUgraphEdgeData*, size_t, CUstreamCaptureMode) = nullptr;
    CUresult (*end)(CUstream, CUgraph*) = nullptr;
#endif
    bool ok() const {
#if CUDA_VERSION >= 12030
        return begin != nullptr && end != nullptr;
#else
        return false;
#endif
    }
};

const Capture& capture() {
    static const Capture c = [] {
        Capture r;
#if CUDA_VERSION >= 12030
#if defined(_WIN32)
        HMODULE h = GetModuleHandleW(L"nvcuda.dll");
        auto sym = [h](const char* n) { return h != nullptr ? (void*) GetProcAddress(h, n) : nullptr; };
#else
        void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
        auto sym = [h](const char* n) { return h != nullptr ? dlsym(h, n) : nullptr; };
#endif
        r.begin = reinterpret_cast<decltype(r.begin)>(sym("cuStreamBeginCaptureToGraph"));
        r.end = reinterpret_cast<decltype(r.end)>(sym("cuStreamEndCapture"));
#endif
        return r;
    }();
    return c;
}

// cuDNN's work (`run`, on `handle`) onto the queue as a native command, as oneMath's cuBLAS backend puts its own: while
// the queue records a SYCL graph, the command's stream is captured into the graph's CUDA graph.  Called on the queue's
// stream straight, the work ran once at the recording and never at the replays (Florence-2's decoder, 2026-11-07).
// False, and nothing submitted, where a recording cannot take it.
bool on_queue(sycl::queue& q, cudnnHandle_t handle, std::function<void()> run) {
    namespace sx = sycl::ext::oneapi::experimental;
    if (q.ext_oneapi_get_state() == sx::queue_state::recording && !capture().ok()) return false;
    q.submit([&](sycl::handler& h) {
        h.ext_codeplay_enqueue_native_command([handle, run](const sycl::interop_handle& ih) {
            std::lock_guard<std::mutex> exec(g_exec_mu);
            const CUstream cs = ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
            if (fe::detail::set_stream(handle, cs) != CUDNN_STATUS_SUCCESS) throw std::runtime_error("cudnnSetStream");
#if CUDA_VERSION >= 12030
            const bool rec = ih.ext_codeplay_has_graph();
            CUgraph g = rec ? ih.ext_codeplay_get_native_graph<sycl::backend::ext_oneapi_cuda>() : nullptr;
            if (rec && capture().begin(cs, g, nullptr, nullptr, 0, CU_STREAM_CAPTURE_MODE_GLOBAL) != CUDA_SUCCESS)
                throw std::runtime_error("cuStreamBeginCaptureToGraph");
            run();
            if (rec && capture().end(cs, &g) != CUDA_SUCCESS) throw std::runtime_error("cuStreamEndCapture");
#else
            run();
#endif
        });
    });
    return true;
}

// Q and O [B][rows][heads * hd], K and V [B][T][kv_heads * hd]; causal: the diagonal at the bottom right (row t sees
// the keys up to T - rows + t), with a window (causal only) the last `window` of them: the band's left bound
std::shared_ptr<fe::graph::Graph> cudnn_graph(cudnnHandle_t handle, int64_t rows, int64_t T, int64_t heads,
                                              int64_t kv_heads, int64_t hd, int64_t B, bool causal, int64_t window) {
    auto g = std::make_shared<fe::graph::Graph>();
    g->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    const int64_t row = heads * hd, kv_row = kv_heads * hd;
    auto Q = g->tensor(fe::graph::Tensor_attributes().set_name("Q").set_uid(kQ).set_dim({B, heads, rows, hd}).set_stride(
        {rows * row, hd, row, 1}));
    auto K = g->tensor(fe::graph::Tensor_attributes().set_name("K").set_uid(kK).set_dim({B, kv_heads, T, hd}).set_stride(
        {T * kv_row, hd, kv_row, 1}));
    auto V = g->tensor(fe::graph::Tensor_attributes().set_name("V").set_uid(kV).set_dim({B, kv_heads, T, hd}).set_stride(
        {T * kv_row, hd, kv_row, 1}));
    auto opts = fe::graph::SDPA_attributes().set_name("sdpa").set_generate_stats(false).set_attn_scale(
        1.0f / std::sqrt((float) hd));
    if (causal) opts.set_causal_mask_bottom_right(true);
    if (causal && window > 0) opts.set_diagonal_band_left_bound((int) window);
    auto [O, stats] = g->sdpa(Q, K, V, opts);
    (void) stats;
    O->set_output(true).set_dim({B, heads, rows, hd}).set_stride({rows * row, hd, row, 1}).set_uid(kO);
    if (!g->build(handle, {fe::HeurMode_t::A}).is_good()) throw std::runtime_error("cuDNN builds no SDPA plan");
    return g;
}

// A training step's graph (no mask): the forward with the softmax's statistics [heads][rows] FP32, or the backward
// from the forward's output, its gradient and the statistics into FP16 gradients; the engine's strides as they are
std::shared_ptr<fe::graph::Graph> train_graph(cudnnHandle_t handle, int64_t rows, int64_t T, int64_t heads, int64_t hd,
                                              bool bwd) {
    auto g = std::make_shared<fe::graph::Graph>();
    g->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    const int64_t row = heads * hd;
    auto t4 = [&](const char* name, int64_t uid, int64_t n) {
        return fe::graph::Tensor_attributes().set_name(name).set_uid(uid).set_dim({1, heads, n, hd}).set_stride(
            {n * row, hd, row, 1});
    };
    auto Q = g->tensor(t4("Q", kQ, rows)), K = g->tensor(t4("K", kK, T)), V = g->tensor(t4("V", kV, T));
    const float scale = 1.0f / std::sqrt((float) hd);
    if (!bwd) {
        auto opts = fe::graph::SDPA_attributes().set_name("sdpa").set_generate_stats(true).set_attn_scale(scale);
        auto [O, stats] = g->sdpa(Q, K, V, opts);
        O->set_output(true).set_dim({1, heads, rows, hd}).set_stride({rows * row, hd, row, 1}).set_uid(kO);
        stats->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_dim({1, heads, rows, 1}).set_stride(
            {heads * rows, rows, 1, 1}).set_uid(kStats);
    } else {
        auto O = g->tensor(t4("O", kO, rows)), dO = g->tensor(t4("dO", kDO, rows));
        auto stats = g->tensor(fe::graph::Tensor_attributes().set_name("Stats").set_uid(kStats).set_dim(
            {1, heads, rows, 1}).set_stride({heads * rows, rows, 1, 1}).set_data_type(fe::DataType_t::FLOAT));
        auto opts = fe::graph::SDPA_backward_attributes().set_name("sdpa_bwd").set_attn_scale(scale);
        auto [dQ, dK, dV] = g->sdpa_backward(Q, K, V, O, dO, stats, opts);
        dQ->set_output(true).set_uid(kDQ).set_dim({1, heads, rows, hd}).set_stride({rows * row, hd, row, 1});
        dK->set_output(true).set_uid(kDK).set_dim({1, heads, T, hd}).set_stride({T * row, hd, row, 1});
        dV->set_output(true).set_uid(kDV).set_dim({1, heads, T, hd}).set_stride({T * row, hd, row, 1});
    }
    if (!g->build(handle, {fe::HeurMode_t::A}).is_good()) throw std::runtime_error("cuDNN builds no SDPA training plan");
    return g;
}

constexpr int64_t kX = 11, kW = 12, kB = 13, kR = 14, kY = 15;   // the convolution's tensors

// x [H][W][C] (FP16, a pixel's stride ldx) by w [O][k][k][C] (FP16), stride 1 and padding k / 2, plus the bias and the
// residual where given, into y [H][W][O] (FP32 or FP16 by y16, a pixel's stride ldy): the bias, the LeakyReLU (`leaky`
// 0 or more: its slope) and the residual (y itself, read and written in the same pass) as the convolution's epilogue
std::shared_ptr<fe::graph::Graph> conv_graph(cudnnHandle_t handle, int64_t H, int64_t W, int64_t C, int64_t O,
                                             int64_t k, bool bias, bool add, int64_t ldx, int64_t ldy, bool y16,
                                             float leaky) {
    auto g = std::make_shared<fe::graph::Graph>();
    g->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto X = g->tensor(fe::graph::Tensor_attributes().set_name("X").set_uid(kX).set_dim({1, C, H, W}).set_stride(
        {H * W * ldx, 1, W * ldx, ldx}));
    auto Wt = g->tensor(fe::graph::Tensor_attributes().set_name("W").set_uid(kW).set_dim({O, C, k, k}).set_stride(
        {k * k * C, 1, k * C, C}));
    const int64_t p = k / 2;
    auto Y = g->conv_fprop(
        X, Wt, fe::graph::Conv_fprop_attributes().set_padding({p, p}).set_stride({1, 1}).set_dilation({1, 1}));
    const std::vector<int64_t> yd{1, O, H, W}, ys{H * W * ldy, 1, W * ldy, ldy};
    if (bias) {
        auto B = g->tensor(fe::graph::Tensor_attributes().set_name("B").set_uid(kB).set_dim({1, O, 1, 1}).set_stride(
            {O, 1, O, O}).set_data_type(fe::DataType_t::FLOAT));
        Y = g->pointwise(Y, B, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD));
    }
    if (leaky >= 0.0f)
        Y = g->pointwise(Y, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD)
                                .set_relu_lower_clip_slope(leaky));
    if (add) {
        auto R = g->tensor(fe::graph::Tensor_attributes().set_name("R").set_uid(kR).set_dim(yd).set_stride(ys)
                               .set_data_type(y16 ? fe::DataType_t::HALF : fe::DataType_t::FLOAT));
        Y = g->pointwise(Y, R, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD));
    }
    Y->set_output(true).set_dim(yd).set_stride(ys).set_uid(kY).set_data_type(y16 ? fe::DataType_t::HALF
                                                                                  : fe::DataType_t::FLOAT);
    // the fallback heuristics too: mode A alone found no plan for these (cuDNN 9 on the RTX 4070, 2026-11-02)
    if (!g->build(handle, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}).is_good())
        throw std::runtime_error("cuDNN builds no convolution plan");
    return g;
}

// x [D][H][W][C] (FP16) by w [O][kt][k][k][C] (FP16), stride 1, padding k / 2 in
// space and none in time, plus the bias and the residual where given, into y [D - kt + 1][H][W][O] (FP32, or FP16 by
// y16)
std::shared_ptr<fe::graph::Graph> conv3_graph(cudnnHandle_t handle, int64_t D, int64_t H, int64_t W, int64_t C,
                                              int64_t O, int64_t kt, int64_t k, bool bias, bool add, bool y16) {
    auto g = std::make_shared<fe::graph::Graph>();
    g->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto X = g->tensor(fe::graph::Tensor_attributes().set_name("X").set_uid(kX).set_dim({1, C, D, H, W}).set_stride(
        {D * H * W * C, 1, H * W * C, W * C, C}));
    auto Wt = g->tensor(fe::graph::Tensor_attributes().set_name("W").set_uid(kW).set_dim({O, C, kt, k, k}).set_stride(
        {kt * k * k * C, 1, k * k * C, k * C, C}));
    const int64_t p = k / 2, Do = D - kt + 1;
    auto Y = g->conv_fprop(X, Wt, fe::graph::Conv_fprop_attributes().set_padding({0, p, p}).set_stride({1, 1, 1})
                                      .set_dilation({1, 1, 1}));
    const std::vector<int64_t> yd{1, O, Do, H, W}, ys{Do * H * W * O, 1, H * W * O, W * O, O};
    if (bias) {
        auto B = g->tensor(fe::graph::Tensor_attributes().set_name("B").set_uid(kB).set_dim({1, O, 1, 1, 1})
                               .set_stride({O, 1, O, O, O}).set_data_type(fe::DataType_t::FLOAT));
        Y = g->pointwise(Y, B, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD));
    }
    if (add) {
        auto R = g->tensor(fe::graph::Tensor_attributes().set_name("R").set_uid(kR).set_dim(yd).set_stride(ys)
                               .set_data_type(y16 ? fe::DataType_t::HALF : fe::DataType_t::FLOAT));
        Y = g->pointwise(Y, R, fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD));
    }
    Y->set_output(true).set_dim(yd).set_stride(ys).set_uid(kY).set_data_type(y16 ? fe::DataType_t::HALF
                                                                                  : fe::DataType_t::FLOAT);
    if (!g->build(handle, {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}).is_good())
        throw std::runtime_error("cuDNN builds no convolution plan");
    return g;
}

// cuDNN's library opened and a handle made: STRATA_CUDNN_LIB, else the system's (libcudnn.so.9; on Windows
// cudnn64_9.dll on the PATH, then NVIDIA's installer's folders, newest first, then the one pip put in the Python the 7z
// carries, nvidia-cudnn-cu12), the first that makes a handle
void open_cudnn(cudnnHandle_t& handle) {
    if (fe::cudnn_dlhandle != nullptr) {
        if (fe::detail::create_handle(&handle) != CUDNN_STATUS_SUCCESS) throw std::runtime_error("cudnnCreate failed");
        return;
    }
    std::vector<std::string> libs;
    if (const char* lib = strata::env_get("CUDNN_LIB")) libs.emplace_back(lib);
#if defined(_WIN32)
    namespace fs = std::filesystem;
    libs.emplace_back("cudnn64_9.dll");
    std::error_code ec;
    std::vector<fs::path> found;
    const char* pf = std::getenv("ProgramFiles");
    const fs::path root = fs::path(pf != nullptr ? pf : "C:\\Program Files") / "NVIDIA" / "CUDNN";
    for (const auto& v : fs::directory_iterator(root, ec))
        for (const auto& b : fs::recursive_directory_iterator(v.path() / "bin", ec))
            if (b.path().filename() == "cudnn64_9.dll") found.push_back(b.path());
    std::sort(found.begin(), found.end(), std::greater<>());
    for (const auto& f : found) libs.push_back(f.string());
    char exe[MAX_PATH] = {};
    fs::path nvidia;   // <7z>\python\Lib\site-packages\nvidia, beside <7z>\app\engine\strata-image.exe
    if (GetModuleFileNameA(nullptr, exe, MAX_PATH) > 0) {
        nvidia = fs::path(exe).parent_path().parent_path().parent_path() / "python" / "Lib" / "site-packages" /
                 "nvidia";
        const fs::path pip = nvidia / "cudnn" / "bin" / "cudnn64_9.dll";
        if (fs::exists(pip, ec)) libs.push_back(pip.string());
    }
    // the CUDA runtime of the engine's CUDA version (cudart64_12.dll), as the user installed it (the PATH, the CUDA
    // toolkit's, the 7z's Python's nvidia-cuda-runtime-cu12), loaded before the first call cudnn-frontend makes through
    // it: a delay-loaded DLL that is missing would end the process at that call
    const std::string cudart = "cudart64_" + std::to_string(CUDART_VERSION / 1000) + ".dll";
    std::vector<fs::path> dirs;
    if (const char* path = std::getenv("PATH")) {
        for (std::string rest = path; !rest.empty();) {
            const size_t k = rest.find(';');
            if (k != 0) dirs.emplace_back(rest.substr(0, k));
            rest = k == std::string::npos ? "" : rest.substr(k + 1);
        }
    }
    if (const char* c = std::getenv("CUDA_PATH")) dirs.push_back(fs::path(c) / "bin");
    if (!nvidia.empty()) dirs.push_back(nvidia / "cuda_runtime" / "bin");
    HMODULE rt = nullptr;
    for (const fs::path& d : dirs)
        if (fs::exists(d / cudart, ec) && (rt = LoadLibraryExA((d / cudart).string().c_str(), nullptr,
                                                               LOAD_WITH_ALTERED_SEARCH_PATH)) != nullptr)
            break;
    if (rt == nullptr && (rt = LoadLibraryA(cudart.c_str())) == nullptr)
        throw std::runtime_error(cudart + ": the CUDA runtime is not installed");
#else
    libs.emplace_back("libcudnn.so.9");
    // cudnn-frontend opens the first of libcudart.so.12 and .13 it finds: the CUDA version the engine was built with
    // (cuDNN's for it on the library path, setup.py's cudnn_lib_dirs), not another a PC also has
    setenv("CUDNN_FRONTEND_CUDART_LIB_NAME", ("libcudart.so." + std::to_string(CUDART_VERSION / 1000)).c_str(), 0);
#endif
    std::string why = "cuDNN is not installed";
    for (const std::string& lib : libs) {
#if defined(_WIN32)
        // its other DLLs beside it: those it links, and those cudnn64_9.dll loads itself later (cudnn_graph64_9.dll and
        // the engines), which Windows looks for on the PATH
        if (lib.find('\\') != std::string::npos) {
            const char* old = std::getenv("PATH");
            const std::string path = fs::path(lib).parent_path().string() + ";" + (old != nullptr ? old : "");
            _putenv_s("PATH", path.c_str());
        }
        HMODULE h = LoadLibraryExA(lib.c_str(), nullptr, lib.find('\\') != std::string::npos ?
                                                             LOAD_WITH_ALTERED_SEARCH_PATH : 0);
#else
        void* h = dlopen(lib.c_str(), RTLD_NOW | RTLD_GLOBAL);
#endif
        if (h == nullptr) continue;
        fe::cudnn_dlhandle = h;
#if defined(_WIN32)
        // through this module's own entry: the delay-loaded imports bind once, to the cudnn64_9.dll loaded then, so
        // they are first called after this one has made a handle (the one kept)
        const auto create = reinterpret_cast<decltype(&cudnnCreate)>(GetProcAddress(h, "cudnnCreate"));
        if (create != nullptr && create(&handle) == CUDNN_STATUS_SUCCESS) return;
#else
        if (fe::detail::create_handle(&handle) == CUDNN_STATUS_SUCCESS) return;
#endif
        why = lib + ": cudnnCreate failed";
#if defined(_WIN32)
        FreeLibrary(h);
#else
        dlclose(h);
#endif
        fe::cudnn_dlhandle = nullptr;
    }
    throw std::runtime_error(why);
}

}  // namespace

namespace {

// The attention of `rows` query rows over T keys (layouts as cudnn_graph's) on q's GPU; false where cuDNN does not
// take it (not opened, no plan: the caller's own kernel), said once by `who`
bool run_sdpa(sycl::queue& q, const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t rows, int64_t T,
              int64_t heads, int64_t kv_heads, int64_t hd, uint16_t* out16, int64_t batch, bool causal,
              int64_t window, const char* who) {
    std::lock_guard<std::mutex> lock(g_mu);
    Cudnn& d = g_cudnn[q.get_device()];
    if (d.failed) return false;
    try {
        if (d.handle == nullptr) open_cudnn(d.handle);
        const auto key = std::make_tuple(rows, T, heads, kv_heads, hd, batch, causal, window);
        auto it = d.graphs.find(key);
        if (it == d.graphs.end()) {
            it = d.graphs.emplace(key, cudnn_graph(d.handle, rows, T, heads, kv_heads, hd, batch, causal, window)).first;
            if (d.graph_lru.size() >= 64) {
                d.graphs.erase(d.graph_lru.back());
                d.graph_lru.pop_back();
            }
        } else {
            d.graph_lru.remove(key);
        }
        d.graph_lru.push_front(key);
        CudnnWs& w = core::Runtime::of_queue(q).queue_item<CudnnWs>(q);
        int64_t ws = 0;
        if (!it->second->get_workspace_size(ws).is_good()) throw std::runtime_error("cuDNN's workspace");
        if (ws > w.ws_bytes) {
            q.wait();   // this queue's work alone reads its workspace
            if (w.ws != nullptr) {
                sycl::free(w.ws, q);
                core::graph_epoch_bump();   // a recorded step may hold the old one
            }
            w.ws = sycl::malloc_device(ws, q);
            w.ws_bytes = w.ws != nullptr ? ws : 0;
        }
        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> pack = {
            {kQ, const_cast<uint16_t*>(q16)}, {kK, const_cast<uint16_t*>(k16)}, {kV, const_cast<uint16_t*>(v16)},
            {kO, out16}};
        return on_queue(q, d.handle, [g = it->second, handle = d.handle, pack, ws = w.ws]() mutable {
            if (!g->execute(handle, pack, ws).is_good()) throw std::runtime_error("cuDNN's SDPA failed");
        });
    } catch (const std::exception& e) {
        platform::downgrade("image.attn.cudnn", std::string(who) + ": cuDNN's attention is not taken on this GPU (" +
                                                e.what() + "): the engine's own kernel");
        d.failed = true;
        return false;
    }
}

}  // namespace

bool cudnn(sycl::queue& q, const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t q0, int64_t q1,
                int64_t T, int64_t heads, int64_t hd, uint16_t* out16, int64_t batch) {
    return run_sdpa(q, q16 + q0 * heads * hd, k16, v16, q1 - q0, T, heads, heads, hd, out16, batch, false, 0,
                    "xestrata-image");
}

bool cudnn_train(sycl::queue& q, bool bwd, const uint16_t* q16, const uint16_t* k16, const uint16_t* v16,
                 const uint16_t* o16, const uint16_t* do16, float* stats, int64_t rows, int64_t keys, int64_t heads,
                 int64_t hd, uint16_t* out16, uint16_t* dq16, uint16_t* dk16, uint16_t* dv16) {
    std::lock_guard<std::mutex> lock(g_mu);
    Cudnn& d = g_cudnn[q.get_device()];
    if (d.failed) return false;
    try {
        if (d.handle == nullptr) open_cudnn(d.handle);
        const auto key = std::make_tuple(rows, keys, heads, hd, bwd);
        auto it = d.train.find(key);
        if (it == d.train.end()) it = d.train.emplace(key, train_graph(d.handle, rows, keys, heads, hd, bwd)).first;
        CudnnWs& w = core::Runtime::of_queue(q).queue_item<CudnnWs>(q);
        int64_t ws = 0;
        if (!it->second->get_workspace_size(ws).is_good()) throw std::runtime_error("cuDNN's workspace");
        if (ws > w.ws_bytes) {
            q.wait();   // this queue's work alone reads its workspace
            if (w.ws != nullptr) {
                sycl::free(w.ws, q);
                core::graph_epoch_bump();
            }
            w.ws = sycl::malloc_device(ws, q);
            w.ws_bytes = w.ws != nullptr ? ws : 0;
        }
        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> pack = {
            {kQ, const_cast<uint16_t*>(q16)}, {kK, const_cast<uint16_t*>(k16)}, {kV, const_cast<uint16_t*>(v16)},
            {kStats, stats}};
        if (!bwd) {
            pack.emplace(kO, out16);
        } else {
            pack.emplace(kO, const_cast<uint16_t*>(o16));
            pack.emplace(kDO, const_cast<uint16_t*>(do16));
            pack.emplace(kDQ, dq16);
            pack.emplace(kDK, dk16);
            pack.emplace(kDV, dv16);
        }
        return on_queue(q, d.handle, [g = it->second, handle = d.handle, pack, ws = w.ws]() mutable {
            if (!g->execute(handle, pack, ws).is_good()) throw std::runtime_error("cuDNN's SDPA training failed");
        });
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata-image: cuDNN's training attention is not taken on this GPU (%s)\n", e.what());
        return false;
    }
}

}  // namespace strata::image::sdpa

namespace strata::image::conv {

bool cudnn(sycl::queue& q, const uint16_t* x, int64_t H, int64_t W, int64_t C, const uint16_t* w, int64_t O, int k,
           const float* bias, bool add, float* y) {
    return cudnn(q, x, H, W, C, w, O, k, bias, add, y, Form{});
}

namespace {

// a convolution's graph for `key` (made by `make` the first time, null where cuDNN takes none) run on q's stream: false
// where cuDNN is not taken (`what` says the shape in the message)
bool run_conv(sycl::queue& q, const std::vector<int64_t>& key,
              const std::function<std::shared_ptr<cudnn_frontend::graph::Graph>(cudnnHandle_t)>& make,
              const uint16_t* x, const uint16_t* w, const float* bias, bool add, void* y, const std::string& what) {
    namespace fe = cudnn_frontend;
    std::lock_guard<std::mutex> lock(sdpa::g_mu);
    sdpa::Cudnn& d = sdpa::g_cudnn[q.get_device()];
    if (d.conv_failed) return false;
    try {
        if (d.handle == nullptr) sdpa::open_cudnn(d.handle);
    } catch (const std::exception& e) {
        platform::downgrade("image.conv.cudnn", std::string("cuDNN's convolution is not taken on this GPU (") + e.what() +
                                                "): the engine's own path");
        d.conv_failed = true;
        return false;
    }
    try {
        auto it = d.convs.find(key);
        if (it == d.convs.end()) it = d.convs.emplace(key, make(d.handle)).first;
        if (!it->second) return false;
        sdpa::CudnnWs& cw = core::Runtime::of_queue(q).queue_item<sdpa::CudnnWs>(q);
        int64_t ws = 0;
        if (!it->second->get_workspace_size(ws).is_good()) throw std::runtime_error("cuDNN's workspace");
        if (ws > cw.ws_bytes) {
            q.wait();   // this queue's work alone reads its workspace
            if (cw.ws != nullptr) {
                sycl::free(cw.ws, q);
                core::graph_epoch_bump();
            }
            cw.ws = sycl::malloc_device(ws, q);
            cw.ws_bytes = cw.ws != nullptr ? ws : 0;
        }
        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> pack = {
            {sdpa::kX, const_cast<uint16_t*>(x)}, {sdpa::kW, const_cast<uint16_t*>(w)}, {sdpa::kY, y}};
        if (bias != nullptr) pack.emplace(sdpa::kB, const_cast<float*>(bias));
        if (add) pack.emplace(sdpa::kR, y);
        return sdpa::on_queue(q, d.handle, [g = it->second, handle = d.handle, pack, ws = cw.ws]() mutable {
            if (!g->execute(handle, pack, ws).is_good()) throw std::runtime_error("cuDNN's convolution failed");
        });
    } catch (const std::exception& e) {   // this shape on the engine's own path from now on
        std::fprintf(stderr,
                     "xestrata-image: cuDNN does not take a convolution of %s (%s): the engine's own path for it\n",
                     what.c_str(), e.what());
        d.convs[key] = nullptr;
        return false;
    }
}

}  // namespace

bool cudnn(sycl::queue& q, const uint16_t* x, int64_t H, int64_t W, int64_t C, const uint16_t* w, int64_t O, int k,
           const float* bias, bool add, void* y, const Form& f) {
    if ((add && f.leaky >= 0.0f) || f.gelu) return false;
    const int64_t lx = f.ldx > 0 ? f.ldx : C, ly = f.ldy > 0 ? f.ldy : O;
    const std::vector<int64_t> key{H, W, C, O, k, bias != nullptr, add, lx, ly, f.y16,
                                   f.leaky >= 0.0f ? (int64_t) (f.leaky * 1e6f) : -1};
    return run_conv(
        q, key,
        [&](cudnnHandle_t h) {
            return sdpa::conv_graph(h, H, W, C, O, k, bias != nullptr, add, lx, ly, f.y16, f.leaky);
        },
        x, w, bias, add, y,
        std::to_string(H) + " x " + std::to_string(W) + " x " + std::to_string(C) + " to " + std::to_string(O));
}

bool cudnn3(sycl::queue& q, const uint16_t* x, int64_t D, int64_t H, int64_t W, int64_t C, const uint16_t* w,
            int64_t O, int kt, int k, const float* bias, bool add, void* y, bool y16) {
    // (the 2D keys have 11 values: these 12 never meet them)
    const std::vector<int64_t> key{3, D, H, W, C, O, kt, k, bias != nullptr, add, y16, 0};
    return run_conv(
        q, key,
        [&](cudnnHandle_t h) { return sdpa::conv3_graph(h, D, H, W, C, O, kt, k, bias != nullptr, add, y16); }, x,
        w, bias, add, y,
        std::to_string(D) + " x " + std::to_string(H) + " x " + std::to_string(W) + " x " + std::to_string(C) + " to " +
            std::to_string(O) + " (" + std::to_string(kt) + " time taps)");
}

}  // namespace strata::image::conv

namespace strata::kernels {

bool cudnn_sdpa(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t rows, int64_t keys, int64_t heads,
                int64_t kv_heads, int64_t hd, uint16_t* out16, void* stream, bool causal, int64_t window) {
    sycl::queue& q = core::Runtime::get().stream(stream);
    if (q.get_device().get_backend() != sycl::backend::ext_oneapi_cuda) return false;
    return image::sdpa::run_sdpa(q, q16, k16, v16, rows, keys, heads, kv_heads, hd, out16, 1, causal, window,
                                 "xestrata");
}

}  // namespace strata::kernels
