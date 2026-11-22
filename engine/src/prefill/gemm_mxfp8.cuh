// src/prefill/gemm_mxfp8.cuh - Clean-room MIT Blackwell (sm_120 / sm_121) cuBLASLt MXFP8 (VEC32_UE8M0) dense prefill.
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// v2 High-Throughput Edition:
//   - Single-slab VRAM bump allocator for cached (W_fp8, SW) weights (0 cudaMalloc calls during prefill!);
//   - Direct GGUF -> MXFP8 fused dequant kernels for Q8_0 and Q6_K (eliminates dequant_f16 intermediate write!);
//   - Shape gate (N >= 1024) skips small-N HC-down projections where X quantization exceeds GEMM savings.
#pragma once

#if !defined(__HIPCC__) && defined(STRATA_CUBLASLT_MXFP8)

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>

namespace strata::prefill::mxfp8 {

__device__ __forceinline__ size_t sf_swizzle_off(int r, int kb, int kb_pad) {
    const int rt = r >> 7;
    const int rm = r & 127;
    const int kt = kb >> 2;
    const int rk = kb & 3;
    const int outer_m = rm & 31;
    const int inner_m = rm >> 5;
    return ((size_t) rt * (size_t) (kb_pad >> 2) + (size_t) kt) * 512u + (size_t) outer_m * 16u +
           (size_t) inner_m * 4u + (size_t) rk;
}

__device__ __forceinline__ void amax_to_ue8m0(float amax, uint8_t& ue8m0, float& inv) {
    if (amax > 0.0f) {
        const float req = amax * (1.0f / 448.0f);
        const uint32_t u = __float_as_uint(req);
        int e = (int) ((u >> 23) & 0xFFu) - 127;
        if ((u & 0x007FFFFFu) != 0u) ++e;
        e = max(-126, min(127, e));
        ue8m0 = (uint8_t) (e + 127);
        inv = __uint_as_float((uint32_t) (127 - e) << 23);
    } else {
        ue8m0 = 1u;
        inv = 0.0f;
    }
}

__device__ __forceinline__ void split_scale_ue8m0(float scale, uint8_t& ue8m0, float& m_d) {
    const float a = fabsf(scale);
    const uint32_t u = __float_as_uint(a);
    const uint32_t exp_b = (u >> 23) & 0xFFu;
    if (exp_b == 0u || exp_b == 255u) {
        ue8m0 = 1u;
        m_d = 0.0f;
        return;
    }
    ue8m0 = (uint8_t) exp_b;
    const uint32_t sign_bit = __float_as_uint(scale) & 0x80000000u;
    m_d = __uint_as_float(sign_bit | (u & 0x007FFFFFu) | 0x3F800000u);
}

// One warp per 32-element block along K for rows [0..rows_pad):
template <bool IS_BF16>
__global__ void __launch_bounds__(256) quant_16_to_mxfp8_kernel(const uint16_t* __restrict__ in, int rows, int rows_pad,
                                                                int K, int nb, int kb_pad,
                                                                uint8_t* __restrict__ out_fp8,
                                                                uint8_t* __restrict__ out_sf) {
    const int64_t w = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int64_t total_blocks = (int64_t) rows_pad * kb_pad;
    if (w >= total_blocks) return;
    const int r = (int) (w / kb_pad);
    const int kb = (int) (w - (int64_t) r * kb_pad);

    if (r >= rows || kb >= nb) {
        if (lane == 0) out_sf[sf_swizzle_off(r, kb, kb_pad)] = 0x7Fu;
        return;
    }

    const size_t idx = (size_t) r * (size_t) K + (size_t) kb * 32u + (size_t) lane;
    const uint16_t raw = __ldg(in + idx);
    const float v = IS_BF16 ? __uint_as_float((uint32_t) raw << 16) : __half2float(__ushort_as_half(raw));

    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));

    uint8_t ue8m0 = 1u;
    float inv = 0.0f;
    amax_to_ue8m0(amax, ue8m0, inv);

    const __nv_fp8_e4m3 q(v * inv);
    out_fp8[idx] = q.__x;
    if (lane == 0) out_sf[sf_swizzle_off(r, kb, kb_pad)] = ue8m0;
}

// Direct Q8_0 (34B per 32 weights) -> MXFP8 (E4M3 + swizzled UE8M0) without FP16 intermediate!
__global__ void __launch_bounds__(256) dequant_q8_0_to_mxfp8_kernel(const uint8_t* __restrict__ w, int rows, int rows_pad,
                                                                    int K, int nb, int kb_pad,
                                                                    uint8_t* __restrict__ out_fp8,
                                                                    uint8_t* __restrict__ out_sf) {
    const int64_t wi = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int64_t total_blocks = (int64_t) rows_pad * kb_pad;
    if (wi >= total_blocks) return;
    const int r = (int) (wi / kb_pad);
    const int kb = (int) (wi - (int64_t) r * kb_pad);

    if (r >= rows || kb >= nb) {
        if (lane == 0) out_sf[sf_swizzle_off(r, kb, kb_pad)] = 0x7Fu;
        return;
    }

    const uint8_t* blk = w + ((size_t) r * (size_t) nb + (size_t) kb) * 34u;
    const uint16_t dh = __ldg(reinterpret_cast<const uint16_t*>(blk));
    const float d = __half2float(__ushort_as_half(dh));
    uint8_t ue8m0 = 1u;
    float m_d = 0.0f;
    split_scale_ue8m0(d, ue8m0, m_d);
    const int8_t qv = static_cast<int8_t>(__ldg(blk + 2 + lane));
    const __nv_fp8_e4m3 fp8(m_d * (float) qv);
    out_fp8[(size_t) r * (size_t) K + (size_t) kb * 32u + (size_t) lane] = fp8.__x;
    if (lane == 0) out_sf[sf_swizzle_off(r, kb, kb_pad)] = ue8m0;
}

// Direct Q6_K (210B per 256 weights = 8 sub-blocks of 32) -> MXFP8 (E4M3 + swizzled UE8M0) without FP16 intermediate!
__global__ void __launch_bounds__(256) dequant_q6_k_to_mxfp8_kernel(const uint8_t* __restrict__ w, int rows, int rows_pad,
                                                                    int K, int nb, int kb_pad,
                                                                    uint8_t* __restrict__ out_fp8,
                                                                    uint8_t* __restrict__ out_sf) {
    const int64_t wi = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int64_t total_blocks = (int64_t) rows_pad * kb_pad;
    if (wi >= total_blocks) return;
    const int r = (int) (wi / kb_pad);
    const int kb = (int) (wi - (int64_t) r * kb_pad);

    if (r >= rows || kb >= nb) {
        if (lane == 0) out_sf[sf_swizzle_off(r, kb, kb_pad)] = 0x7Fu;
        return;
    }

    const int nsb = nb >> 3;
    const int sb = kb >> 3;
    const int ib = kb & 7;      // 0..7 sub-block of 32 inside 256-val Q6_K superblock
    const uint8_t* blk = w + ((size_t) r * (size_t) nsb + (size_t) sb) * 210u;
    const uint16_t dh = __ldg(reinterpret_cast<const uint16_t*>(blk + 208));
    const float d_sup = __half2float(__ushort_as_half(dh));

    // Q6_K layout: two 128-val halves (n = ib >> 2); within a 128-val half, 4 sub-blocks of 32 (q = ib & 3):
    //   q=0: low nibbles of ql[0..31],  qh bits 0..1
    //   q=1: low nibbles of ql[32..63], qh bits 2..3
    //   q=2: high nibbles of ql[0..31], qh bits 4..5
    //   q=3: high nibbles of ql[32..63],qh bits 6..7
    const int n = ib >> 2;
    const int q = ib & 3;
    const uint8_t* ql = blk + 64 * n + 32 * (q & 1);
    const uint8_t* qh = blk + 128 + 32 * n;
    const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192 + 8 * n + 2 * q);

    const uint8_t ql_b = __ldg(ql + lane);
    const uint8_t qh_b = __ldg(qh + lane);
    const int low4 = (q < 2) ? (ql_b & 0xFu) : (ql_b >> 4);
    const int hi2 = (qh_b >> (2 * q)) & 3;
    const int q6 = (low4 | (hi2 << 4)) - 32;

    const float s0 = d_sup * (float) __ldg(sc);
    const float s1 = d_sup * (float) __ldg(sc + 1);
    const float val = ((lane < 16) ? s0 : s1) * (float) q6;

    // Exact power-of-2 scale from the two 16-element sub-scales (max |q6| = 32 -> 32 * max(|s0|, |s1|)).
    const float amax = 32.0f * fmaxf(fabsf(s0), fabsf(s1));
    uint8_t ue8m0 = 1u;
    float inv = 0.0f;
    amax_to_ue8m0(amax, ue8m0, inv);

    const __nv_fp8_e4m3 fp8(val * inv);
    out_fp8[(size_t) r * (size_t) K + (size_t) kb * 32u + (size_t) lane] = fp8.__x;
    if (lane == 0) out_sf[sf_swizzle_off(r, kb, kb_pad)] = ue8m0;
}

struct ShapeKey {
    int m, n, k, ldc;
    bool operator==(const ShapeKey& o) const { return m == o.m && n == o.n && k == o.k && ldc == o.ldc; }
};

struct ShapeKeyHash {
    size_t operator()(const ShapeKey& s) const {
        size_t h = (size_t) s.m * 2654435761u;
        h ^= (size_t) s.n * 2246822519u + 0x9e3779b9u + (h << 6) + (h >> 2);
        h ^= (size_t) s.k * 3266489917u + 0x9e3779b9u + (h << 6) + (h >> 2);
        h ^= (size_t) s.ldc * 668265263u + 0x9e3779b9u + (h << 6) + (h >> 2);
        return h;
    }
};

struct CachedAlgo {
    bool ok = false;
    cublasLtMatmulAlgo_t algo{};
};

struct CachedWeight {
    const void* src_ptr = nullptr;
    int type = -1;
    int64_t n = 0;
    int64_t k = 0;
    uint8_t* fp8 = nullptr;
    uint8_t* sf = nullptr;
};

struct State {
    cublasLtHandle_t lt = nullptr;
    void* ws = nullptr;
    size_t ws_bytes = 0;
    bool own_ws = false;

    uint8_t* x_fp8 = nullptr;
    size_t x_fp8_cap = 0;
    uint8_t* x_sf = nullptr;
    size_t x_sf_cap = 0;

    uint8_t* w_fp8 = nullptr;
    size_t w_fp8_cap = 0;
    uint8_t* w_sf = nullptr;
    size_t w_sf_cap = 0;

    // Single pre-allocated slab for cached W_fp8 + SW -> ZERO cudaMalloc/cudaMemGetInfo calls during prefill!
    uint8_t* slab = nullptr;
    size_t slab_cap = 0;
    size_t slab_used = 0;
    bool slab_tried = false;

    std::unordered_map<ShapeKey, CachedAlgo, ShapeKeyHash> algos;
    std::vector<CachedWeight> wcache;
    bool failed = false;
    bool logged = false;

    ~State() {
        if (slab) cudaFree(slab);
        if (x_fp8) cudaFree(x_fp8);
        if (x_sf) cudaFree(x_sf);
        if (w_fp8) cudaFree(w_fp8);
        if (w_sf) cudaFree(w_sf);
        if (own_ws && ws) cudaFree(ws);
        if (lt) cublasLtDestroy(lt);
    }

    static bool grow_buf(uint8_t*& p, size_t& cap, size_t need) {
        if (need <= cap && p != nullptr) return true;
        if (p != nullptr) { cudaFree(p); p = nullptr; cap = 0; }
        const size_t alloc = (need + 65535u) & ~(size_t) 65535u;
        if (cudaMalloc((void**) &p, alloc) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        cap = alloc;
        return true;
    }

    bool init(void* workspace, size_t workspace_bytes) {
        if (lt != nullptr) return !failed;
        if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) {
            failed = true;
            return false;
        }
        ws = workspace;
        ws_bytes = workspace_bytes;
        if (ws == nullptr || ws_bytes < (4u << 20)) {
            const size_t want = 32u << 20;
            if (cudaMalloc(&ws, want) == cudaSuccess) {
                ws_bytes = want;
                own_ws = true;
            } else {
                (void) cudaGetLastError();
                ws = workspace;
                ws_bytes = workspace_bytes;
            }
        }
        if (!slab_tried) {
            slab_tried = true;
            int64_t budget = 3072ll << 20;   // 3 GiB default slab if card has >= 7 GiB free VRAM
            if (const char* v = std::getenv("STRATA_PF_MXFP8_CACHE_MIB"); v != nullptr && v[0] != '\0') {
                budget = (int64_t) std::atoll(v) * (1ll << 20);
            }
            if (budget > 0) {
                size_t fb = 0, tb = 0;
                const size_t reserve = 4ull << 30;
                if (cudaMemGetInfo(&fb, &tb) == cudaSuccess && fb > reserve + (256ull << 20)) {
                    const size_t want = std::min<size_t>((size_t) budget, fb - reserve);
                    if (cudaMalloc((void**) &slab, want) == cudaSuccess) {
                        slab_cap = want;
                        slab_used = 0;
                    } else {
                        (void) cudaGetLastError();
                    }
                }
            }
        }
        return true;
    }

    void rebind_ws(void* workspace, size_t workspace_bytes) {
        if (!own_ws) {
            ws = workspace;
            ws_bytes = workspace_bytes;
        }
    }

    bool quantize_16(const uint16_t* src, bool is_bf16, int rows, int K, uint8_t* dst_fp8, uint8_t* dst_sf,
                     cudaStream_t s) {
        const int nb = K >> 5;
        const int kb_pad = (nb + 3) & ~3;
        const int rows_pad = (rows + 127) & ~127;
        const int64_t total_blocks = (int64_t) rows_pad * kb_pad;
        const unsigned grid = (unsigned) ((total_blocks + 7) / 8);
        if (is_bf16) {
            quant_16_to_mxfp8_kernel<true><<<grid, 256, 0, s>>>(src, rows, rows_pad, K, nb, kb_pad, dst_fp8, dst_sf);
        } else {
            quant_16_to_mxfp8_kernel<false><<<grid, 256, 0, s>>>(src, rows, rows_pad, K, nb, kb_pad, dst_fp8, dst_sf);
        }
        return cudaGetLastError() == cudaSuccess;
    }

    bool direct_dequant_to_mxfp8(int ggml_type, const void* w_blocks, int rows, int K, uint8_t* dst_fp8,
                                 uint8_t* dst_sf, cudaStream_t s) {
        if (ggml_type != 8 && ggml_type != 14) return false;
        if (ggml_type == 14 && (K & 255) != 0) return false;
        const int nb = K >> 5;
        const int kb_pad = (nb + 3) & ~3;
        const int rows_pad = (rows + 127) & ~127;
        const int64_t total_blocks = (int64_t) rows_pad * kb_pad;
        const unsigned grid = (unsigned) ((total_blocks + 7) / 8);
        if (ggml_type == 8) {
            dequant_q8_0_to_mxfp8_kernel<<<grid, 256, 0, s>>>(static_cast<const uint8_t*>(w_blocks), rows, rows_pad, K,
                                                              nb, kb_pad, dst_fp8, dst_sf);
        } else {
            dequant_q6_k_to_mxfp8_kernel<<<grid, 256, 0, s>>>(static_cast<const uint8_t*>(w_blocks), rows, rows_pad, K,
                                                              nb, kb_pad, dst_fp8, dst_sf);
        }
        return cudaGetLastError() == cudaSuccess;
    }

    bool find_cached(const void* src_ptr, int type, int64_t N, int64_t K, uint8_t*& out_fp8, uint8_t*& out_sf) const {
        for (const auto& e : wcache) {
            if (e.src_ptr == src_ptr && e.type == type && e.n == N && e.k == K) {
                out_fp8 = e.fp8;
                out_sf = e.sf;
                return true;
            }
        }
        return false;
    }

    bool alloc_from_slab(size_t fp8_bytes, size_t sf_bytes, uint8_t*& p_fp8, uint8_t*& p_sf) {
        if (slab == nullptr) return false;
        const size_t a_fp8 = (fp8_bytes + 255u) & ~(size_t) 255u;
        const size_t a_sf = (sf_bytes + 255u) & ~(size_t) 255u;
        if (slab_used + a_fp8 + a_sf > slab_cap) return false;
        p_fp8 = slab + slab_used;
        p_sf = slab + slab_used + a_fp8;
        slab_used += a_fp8 + a_sf;
        return true;
    }

    bool matmul_fp8(const uint8_t* dW_fp8, const uint8_t* dSW, const uint8_t* dX_fp8, const uint8_t* dSX,
                    float* Y, int M, int N_tok, int K, int ldc, float beta, cudaStream_t s) {
        if (failed || lt == nullptr) return false;
        cublasLtMatmulDesc_t desc = nullptr;
        cublasLtMatrixLayout_t lA = nullptr, lB = nullptr, lC = nullptr;
        cublasStatus_t st = cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        if (st != CUBLAS_STATUS_SUCCESS) return false;

        const cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
        const cublasLtMatmulMatrixScale_t sm = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        const void* pSA = dSW;
        const void* pSB = dSX;
        st = cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        if (st == CUBLAS_STATUS_SUCCESS)
            st = cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        if (st == CUBLAS_STATUS_SUCCESS)
            st = cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm));
        if (st == CUBLAS_STATUS_SUCCESS)
            st = cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm));
        if (st == CUBLAS_STATUS_SUCCESS)
            st = cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &pSA, sizeof(pSA));
        if (st == CUBLAS_STATUS_SUCCESS)
            st = cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &pSB, sizeof(pSB));

        if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&lA, CUDA_R_8F_E4M3, K, M, K);
        if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&lB, CUDA_R_8F_E4M3, K, N_tok, K);
        if (st == CUBLAS_STATUS_SUCCESS) st = cublasLtMatrixLayoutCreate(&lC, CUDA_R_32F, M, N_tok, ldc);

        bool ok = false;
        if (st == CUBLAS_STATUS_SUCCESS) {
            const ShapeKey key{M, N_tok, K, ldc};
            auto it = algos.find(key);
            if (it == algos.end()) {
                CachedAlgo ca{};
                cublasLtMatmulPreference_t pref = nullptr;
                if (cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS) {
                    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes,
                                                         sizeof(ws_bytes));
                    cublasLtMatmulHeuristicResult_t hr{};
                    int nres = 0;
                    if (cublasLtMatmulAlgoGetHeuristic(lt, desc, lA, lB, lC, lC, pref, 1, &hr, &nres) ==
                            CUBLAS_STATUS_SUCCESS &&
                        nres > 0) {
                        ca.ok = true;
                        ca.algo = hr.algo;
                    }
                    cublasLtMatmulPreferenceDestroy(pref);
                }
                it = algos.emplace(key, ca).first;
            }
            if (it->second.ok) {
                const float alpha = 1.0f;
                const cublasStatus_t ms = cublasLtMatmul(lt, desc, &alpha, dW_fp8, lA, dX_fp8, lB, &beta, Y, lC, Y, lC,
                                                         &it->second.algo, ws, ws_bytes, s);
                ok = (ms == CUBLAS_STATUS_SUCCESS);
            }
        }
        if (lC) cublasLtMatrixLayoutDestroy(lC);
        if (lB) cublasLtMatrixLayoutDestroy(lB);
        if (lA) cublasLtMatrixLayoutDestroy(lA);
        if (desc) cublasLtMatmulDescDestroy(desc);
        return ok;
    }

    bool run_16(const uint16_t* X, bool x_is_bf16, const void* w_key, int w_type, const uint16_t* w16_ready,
                bool w_is_bf16, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy, float beta, cudaStream_t s,
                bool (*dequant_fn)(void*, uint16_t*), void* dequant_ctx, uint16_t* dequant_buf) {
        // Gate N >= 1024 and T >= min_t (default 4096 where MXFP8 compute beats FP16+side-dequant; warm cache hits run at T >= 512).
        static const int64_t min_t = [] {
            const char* v = std::getenv("STRATA_PF_MXFP8_MIN_T");
            return (v != nullptr && v[0] != '\0') ? (int64_t) std::atoll(v) : (int64_t) 4096;
        }();
        uint8_t* p_wfp8 = nullptr;
        uint8_t* p_wsf = nullptr;
        const bool hit = find_cached(w_key, w_type, N, K, p_wfp8, p_wsf);
        if (failed || T < (hit ? 512 : min_t) || N < 1024 || K < 256 || (N & 15) != 0 || (K & 31) != 0) return false;
        const int64_t T_al = T & ~15ll;
        if (T_al < 256) return false;
        const int ldc = (int) (ldy > 0 ? ldy : N);
        if ((ldc & 3) != 0) return false;

        const int nb = (int) (K >> 5);
        const int kb_pad = (nb + 3) & ~3;
        const int t_pad = (int) ((T_al + 127) & ~127);
        const int n_pad = (int) ((N + 127) & ~127);

        if (!grow_buf(x_fp8, x_fp8_cap, (size_t) T_al * (size_t) K) ||
            !grow_buf(x_sf, x_sf_cap, (size_t) t_pad * (size_t) kb_pad)) {
            return false;
        }
        if (!quantize_16(X, x_is_bf16, (int) T_al, (int) K, x_fp8, x_sf, s)) return false;

        if (!hit) {
            const size_t fp8_bytes = (size_t) N * (size_t) K;
            const size_t sf_bytes = (size_t) n_pad * (size_t) kb_pad;
            const bool cached = (w_key != nullptr) && alloc_from_slab(fp8_bytes, sf_bytes, p_wfp8, p_wsf);
            if (!cached) {
                if (!grow_buf(w_fp8, w_fp8_cap, fp8_bytes) || !grow_buf(w_sf, w_sf_cap, sf_bytes)) return false;
                p_wfp8 = w_fp8;
                p_wsf = w_sf;
            }
            bool ok_w = false;
            if (w16_ready != nullptr) {
                ok_w = quantize_16(w16_ready, w_is_bf16, (int) N, (int) K, p_wfp8, p_wsf, s);
            } else if (direct_dequant_to_mxfp8(w_type, w_key, (int) N, (int) K, p_wfp8, p_wsf, s)) {
                ok_w = true;
            } else if (dequant_fn != nullptr && dequant_buf != nullptr && dequant_fn(dequant_ctx, dequant_buf)) {
                ok_w = quantize_16(dequant_buf, false, (int) N, (int) K, p_wfp8, p_wsf, s);
            }
            if (!ok_w) return false;
            if (cached) wcache.push_back({w_key, w_type, N, K, p_wfp8, p_wsf});
        }

        if (!matmul_fp8(p_wfp8, p_wsf, x_fp8, x_sf, Y, (int) N, (int) T_al, (int) K, ldc, beta, s)) {
            return false;
        }
        if (!logged) {
            logged = true;
            std::fprintf(stderr, "strata prefill: Blackwell cuBLASLt MXFP8 (VEC32_UE8M0) dense GEMM active "
                                 "(slab %zu MiB, STRATA_PF_MXFP8=0 disables)\n",
                         slab_cap >> 20);
        }
        return true;
    }
};

inline bool enabled() {
    static const int env = [] {
        const char* v = std::getenv("STRATA_PF_MXFP8");
        if (v == nullptr || v[0] == '\0') return 0;   // opt-in: E4M3 3-bit mantissa can perturb greedy logits on Q8_0/BF16 dense layers
        return std::atoi(v) != 0 ? 1 : 0;
    }();
    if (env == 0) return false;
    static int dev_ok[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) {
        (void) cudaGetLastError();
        return false;
    }
    if (dev_ok[dev] == 0) {
        int ma = 0;
        const bool ok = cudaDeviceGetAttribute(&ma, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess && ma >= 12;
        if (!ok) (void) cudaGetLastError();
        dev_ok[dev] = ok ? 1 : -1;
    }
    return dev_ok[dev] > 0;
}

}  // namespace strata::prefill::mxfp8

#endif
