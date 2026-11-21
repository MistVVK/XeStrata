// engine/src/inkling2/hip/ink_hip_blaslt.hpp
// Dynamic zero-link-dependency hipBLASLt wrapper for high-TFLOP Tensile BF16/FP16 GEMM on AMD RDNA3/3.5/4.
// Loads libhipblaslt.{dll,so.1,so} at runtime; cleanly falls back to native Wave32 WMMA/fdot2 kernels if unavailable.
// MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors.
#pragma once

#include "strata/hip_uma_guard.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace strata::inkling2::hip {

// ABI-exact minimal C definitions matching ROCm hipblaslt.h + hip/library_types.h + hipblas-common.h
struct DynBlasLtAlgo {
    uint8_t data[16]{};
    size_t max_workspace_bytes = 0;
};

struct DynBlasLtHeurResult {
    DynBlasLtAlgo algo{};
    size_t workspaceSize = 0;
    int32_t state = 0;
    float wavesCount = 1.0f;
    int reserved[4]{};
};

class HipBlasLtRunner {
public:
    static constexpr int kTypeF32  = 0;   // HIP_R_32F
    static constexpr int kTypeF16  = 2;   // HIP_R_16F
    static constexpr int kTypeBF16 = 14;  // HIP_R_16BF
    static constexpr int kOpN      = 111; // HIPBLAS_OP_N
    static constexpr int kOpT      = 112; // HIPBLAS_OP_T
    static constexpr int kCompF32  = 2;   // HIPBLAS_COMPUTE_32F
    static constexpr int kAttrTransA = 0;
    static constexpr int kAttrTransB = 1;
    static constexpr int kPrefMaxWs  = 1;

    ~HipBlasLtRunner() { release(); }

    bool init(bool is_uma, strata::hip_uma::AmdGen gen) noexcept {
        release();
        if (gen == strata::hip_uma::AmdGen::Rdna2 || gen == strata::hip_uma::AmdGen::Unknown) {
            return false;
        }
        // Default ON for RDNA4 (gfx120x, tuned ROCm 7 Tensile BF16->FP32); default OFF for RDNA3/3.5
        // (native Wave32 k_bf16_wmma_bf16_64x128 + fdot2<8> beats Windows/ROCm gfx1151 Tensile by +26-42%).
        // Override explicitly via STRATA_HIPBLASLT=1 or STRATA_HIPBLASLT=0.
        bool enable = (gen == strata::hip_uma::AmdGen::Rdna4);
        if (const char* env = std::getenv("STRATA_HIPBLASLT")) {
            enable = (std::atoi(env) != 0);
        }
        if (!enable) return false;

#ifdef _WIN32
        lib_ = static_cast<void*>(LoadLibraryA("libhipblaslt.dll"));
        if (!lib_) lib_ = static_cast<void*>(LoadLibraryA("hipblaslt.dll"));
#else
        lib_ = dlopen("libhipblaslt.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (!lib_) lib_ = dlopen("libhipblaslt.so", RTLD_LAZY | RTLD_LOCAL);
#endif
        if (!lib_) return false;

        auto sym = [&](const char* name) -> void* {
#ifdef _WIN32
            return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(lib_), name));
#else
            return dlsym(lib_, name);
#endif
        };

        fn_create_           = reinterpret_cast<FnCreate>(sym("hipblasLtCreate"));
        fn_destroy_          = reinterpret_cast<FnDestroy>(sym("hipblasLtDestroy"));
        fn_md_create_        = reinterpret_cast<FnMdCreate>(sym("hipblasLtMatmulDescCreate"));
        fn_md_destroy_       = reinterpret_cast<FnMdDestroy>(sym("hipblasLtMatmulDescDestroy"));
        fn_md_set_           = reinterpret_cast<FnMdSet>(sym("hipblasLtMatmulDescSetAttribute"));
        fn_ml_create_        = reinterpret_cast<FnMlCreate>(sym("hipblasLtMatrixLayoutCreate"));
        fn_ml_destroy_       = reinterpret_cast<FnMlDestroy>(sym("hipblasLtMatrixLayoutDestroy"));
        fn_pref_create_      = reinterpret_cast<FnPrefCreate>(sym("hipblasLtMatmulPreferenceCreate"));
        fn_pref_destroy_     = reinterpret_cast<FnPrefDestroy>(sym("hipblasLtMatmulPreferenceDestroy"));
        fn_pref_set_         = reinterpret_cast<FnPrefSet>(sym("hipblasLtMatmulPreferenceSetAttribute"));
        fn_get_heur_         = reinterpret_cast<FnGetHeur>(sym("hipblasLtMatmulAlgoGetHeuristic"));
        fn_matmul_           = reinterpret_cast<FnMatmul>(sym("hipblasLtMatmul"));

        if (!fn_create_ || !fn_destroy_ || !fn_md_create_ || !fn_md_destroy_ || !fn_md_set_ ||
            !fn_ml_create_ || !fn_ml_destroy_ || !fn_pref_create_ || !fn_pref_destroy_ ||
            !fn_pref_set_ || !fn_get_heur_ || !fn_matmul_) {
            release();
            return false;
        }
        if (fn_create_(&handle_) != 0 || !handle_) {
            release();
            return false;
        }
        if (fn_pref_create_(&pref_) != 0 || !pref_) {
            release();
            return false;
        }
        ws_bytes_ = 32ULL << 20;
        if (strata::hip_uma::alloc_gpu_buffer(ws_, ws_bytes_, is_uma) != hipSuccess) {
            ws_bytes_ = 8ULL << 20;
            if (strata::hip_uma::alloc_gpu_buffer(ws_, ws_bytes_, is_uma) != hipSuccess) {
                ws_bytes_ = 0;
            }
        }
        const uint64_t max_ws = static_cast<uint64_t>(ws_bytes_);
        (void)fn_pref_set_(pref_, kPrefMaxWs, &max_ws, sizeof(max_ws));
        ready_ = true;
        return true;
    }

    bool ready() const noexcept { return ready_; }

    // One-time warmup inside backend init(): pays the ~600ms lazy Tensile library load at startup
    // instead of the user's first prompt TTFT, and pre-caches common trunk (M, N, K) shapes!
    void warmup(bool is_uma, hipStream_t stream) noexcept {
        if (!ready_) return;
        strata::hip_uma::GpuBuffer da, db, dc;
        if (strata::hip_uma::alloc_gpu_buffer(da, 64u * 64u * 2u, is_uma) == hipSuccess &&
            strata::hip_uma::alloc_gpu_buffer(db, 64u * 64u * 2u, is_uma) == hipSuccess &&
            strata::hip_uma::alloc_gpu_buffer(dc, 64u * 64u * 4u, is_uma) == hipSuccess) {
            (void)hipMemsetAsync(da.dev_ptr, 0, da.bytes, stream);
            (void)hipMemsetAsync(db.dev_ptr, 0, db.bytes, stream);
            (void)gemm_bf16_f32(reinterpret_cast<const uint16_t*>(da.dev_ptr),
                                reinterpret_cast<const uint16_t*>(db.dev_ptr),
                                reinterpret_cast<float*>(dc.dev_ptr),
                                64, 64, 64, stream, 1.0f);
            (void)hipStreamSynchronize(stream);
        }
        strata::hip_uma::free_gpu_buffer(da);
        strata::hip_uma::free_gpu_buffer(db);
        strata::hip_uma::free_gpu_buffer(dc);
    }

    // Row-major Y[M, N] = alpha * (X[M, K] @ W[N, K]^T) via column-major Y^T[N, M] = alpha * ((W^T)^T[N, K] @ X^T[K, M])
    bool gemm_tn(int in_type, int out_type,
                 const void* W_nk, const void* X_mk, void* Y_mn,
                 int M, int N, int K, hipStream_t stream, float alpha = 1.0f) noexcept {
        if (!ready_ || M <= 0 || N <= 0 || K <= 0) return false;
        const PlanKey key{in_type, out_type, M, N, K};
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            CachedPlan cp{};
            if (build_plan(key, cp)) {
                it = cache_.emplace(key, cp).first;
            } else {
                cp.valid = false;
                it = cache_.emplace(key, cp).first;
            }
        }
        const CachedPlan& plan = it->second;
        if (!plan.valid) return false;

        const float beta = 0.0f;
        const int32_t st = fn_matmul_(
            handle_, plan.md,
            &alpha,
            W_nk, plan.ad,
            X_mk, plan.bd,
            &beta,
            Y_mn, plan.cd,
            Y_mn, plan.dd,
            &plan.algo,
            ws_.dev_ptr, ws_bytes_,
            stream);
        return st == 0;
    }

    bool gemm_bf16_f32(const uint16_t* W_nk, const uint16_t* X_mk, float* Y_mn,
                       int M, int N, int K, hipStream_t stream, float alpha = 1.0f) noexcept {
        return gemm_tn(kTypeBF16, kTypeF32, W_nk, X_mk, Y_mn, M, N, K, stream, alpha);
    }

    bool gemm_f16_f32(const void* W_nk, const void* X_mk, float* Y_mn,
                      int M, int N, int K, hipStream_t stream, float alpha = 1.0f) noexcept {
        return gemm_tn(kTypeF16, kTypeF32, W_nk, X_mk, Y_mn, M, N, K, stream, alpha);
    }

    void release() noexcept {
        for (auto& kv : cache_) {
            const auto& cp = kv.second;
            if (cp.ad && fn_ml_destroy_) fn_ml_destroy_(cp.ad);
            if (cp.bd && fn_ml_destroy_) fn_ml_destroy_(cp.bd);
            if (cp.cd && fn_ml_destroy_) fn_ml_destroy_(cp.cd);
            if (cp.dd && fn_ml_destroy_) fn_ml_destroy_(cp.dd);
            if (cp.md && fn_md_destroy_) fn_md_destroy_(cp.md);
        }
        cache_.clear();
        if (pref_ && fn_pref_destroy_) { fn_pref_destroy_(pref_); pref_ = nullptr; }
        if (handle_ && fn_destroy_) { fn_destroy_(handle_); handle_ = nullptr; }
        strata::hip_uma::free_gpu_buffer(ws_);
        ws_bytes_ = 0;
        if (lib_) {
#ifdef _WIN32
            FreeLibrary(static_cast<HMODULE>(lib_));
#else
            dlclose(lib_);
#endif
            lib_ = nullptr;
        }
        ready_ = false;
    }

private:
    struct PlanKey {
        int in_type, out_type, M, N, K;
        bool operator==(const PlanKey& o) const noexcept {
            return in_type == o.in_type && out_type == o.out_type && M == o.M && N == o.N && K == o.K;
        }
    };
    struct PlanKeyHash {
        size_t operator()(const PlanKey& k) const noexcept {
            size_t h = static_cast<size_t>(k.in_type) * 1315423911u;
            h ^= static_cast<size_t>(k.out_type) + 0x9e3779b9u + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(k.M) + 0x9e3779b9u + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(k.N) + 0x9e3779b9u + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(k.K) + 0x9e3779b9u + (h << 6) + (h >> 2);
            return h;
        }
    };
    struct CachedPlan {
        bool valid = false;
        void* md = nullptr;
        void* ad = nullptr;
        void* bd = nullptr;
        void* cd = nullptr;
        void* dd = nullptr;
        DynBlasLtAlgo algo{};
    };

    bool build_plan(const PlanKey& k, CachedPlan& out) noexcept {
        if (fn_md_create_(&out.md, kCompF32, kTypeF32) != 0) return false;
        const int32_t ta = kOpT, tb = kOpN;
        if (fn_md_set_(out.md, kAttrTransA, &ta, sizeof(ta)) != 0 ||
            fn_md_set_(out.md, kAttrTransB, &tb, sizeof(tb)) != 0) {
            return false;
        }
        // Column-major A = W^T [K, N] (transA=T -> [N, K]), B = X^T [K, M], C=D = Y^T [N, M]
        if (fn_ml_create_(&out.ad, k.in_type,  static_cast<uint64_t>(k.K), static_cast<uint64_t>(k.N), static_cast<int64_t>(k.K)) != 0 ||
            fn_ml_create_(&out.bd, k.in_type,  static_cast<uint64_t>(k.K), static_cast<uint64_t>(k.M), static_cast<int64_t>(k.K)) != 0 ||
            fn_ml_create_(&out.cd, k.out_type, static_cast<uint64_t>(k.N), static_cast<uint64_t>(k.M), static_cast<int64_t>(k.N)) != 0 ||
            fn_ml_create_(&out.dd, k.out_type, static_cast<uint64_t>(k.N), static_cast<uint64_t>(k.M), static_cast<int64_t>(k.N)) != 0) {
            return false;
        }
        DynBlasLtHeurResult res[4]{};
        int returned = 0;
        if (fn_get_heur_(handle_, out.md, out.ad, out.bd, out.cd, out.dd, pref_, 4, res, &returned) != 0 || returned <= 0) {
            return false;
        }
        for (int i = 0; i < returned; ++i) {
            if (res[i].state == 0 && res[i].workspaceSize <= ws_bytes_) {
                out.algo = res[i].algo;
                out.valid = true;
                return true;
            }
        }
        return false;
    }

    using FnCreate      = int32_t (*)(void**);
    using FnDestroy     = int32_t (*)(void*);
    using FnMdCreate    = int32_t (*)(void**, int, int);
    using FnMdDestroy   = int32_t (*)(void*);
    using FnMdSet       = int32_t (*)(void*, int, const void*, size_t);
    using FnMlCreate    = int32_t (*)(void**, int, uint64_t, uint64_t, int64_t);
    using FnMlDestroy   = int32_t (*)(void*);
    using FnPrefCreate  = int32_t (*)(void**);
    using FnPrefDestroy = int32_t (*)(void*);
    using FnPrefSet     = int32_t (*)(void*, int, const void*, size_t);
    using FnGetHeur     = int32_t (*)(void*, void*, void*, void*, void*, void*, void*, int, DynBlasLtHeurResult*, int*);
    using FnMatmul      = int32_t (*)(void*, void*, const void*, const void*, void*, const void*, void*,
                                      const void*, const void*, void*, void*, void*, const DynBlasLtAlgo*,
                                      void*, size_t, hipStream_t);

    void* lib_ = nullptr;
    void* handle_ = nullptr;
    void* pref_ = nullptr;
    strata::hip_uma::GpuBuffer ws_{};
    size_t ws_bytes_ = 0;
    bool ready_ = false;
    std::unordered_map<PlanKey, CachedPlan, PlanKeyHash> cache_;

    FnCreate      fn_create_       = nullptr;
    FnDestroy     fn_destroy_      = nullptr;
    FnMdCreate    fn_md_create_    = nullptr;
    FnMdDestroy   fn_md_destroy_   = nullptr;
    FnMdSet       fn_md_set_       = nullptr;
    FnMlCreate    fn_ml_create_    = nullptr;
    FnMlDestroy   fn_ml_destroy_   = nullptr;
    FnPrefCreate  fn_pref_create_  = nullptr;
    FnPrefDestroy fn_pref_destroy_ = nullptr;
    FnPrefSet     fn_pref_set_     = nullptr;
    FnGetHeur     fn_get_heur_     = nullptr;
    FnMatmul      fn_matmul_       = nullptr;
};

}  // namespace strata::inkling2::hip
