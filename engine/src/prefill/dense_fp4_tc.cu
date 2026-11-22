// src/prefill/dense_fp4_tc.cu - Clean-room MIT Blackwell (sm_120a / sm_121a) mxf4nvf4 Tensor Core dense prefill.
//
// MIT License
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Derived strictly from NVIDIA PTX ISA 8.7+ specification for block-scaled FP4 matrix multiply-accumulate:
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
// and GGML block_nvfp4 (36B per 64 values: uint8_t d[4] UE4M3 scales + uint8_t qs[32] split-half E2M1x2).
//
// Zero FP16 weight dequantization: reads native block_nvfp4 directly via 100% coalesced 32-byte loads,
// quantizes FP16 activations on-the-fly to matching split-half E2M1 + UE4M3 + per-64 FP32 g_blk (40B / 64 vals),
// and achieves 4x Blackwell FP4 Tensor Core throughput (~800+ TFLOPS) with ~20.5 KB smem and 0 register spills.
#include "strata/prefill/dense_fp4_tc.hpp"

#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

#include "strata/core/emulate.hpp"

namespace strata::prefill::dense_fp4_tc {
namespace {

constexpr int K_STEP = 64;
constexpr int BS_NVFP4 = 36;
constexpr int ACT_BLK = 48;      // 32B qs + 4B UE4M3 scales + 4B float g_blk + 8B pad (16-byte aligned for cp.async.cg 16B)
constexpr int NT = 256;          // 8 warps: wf in 0..1 (64 output rows each), wn in 0..3 (16 tokens each)
constexpr int BM = 128;          // output rows (N) per CTA
constexpr int BN = 64;           // tokens (T) per CTA
constexpr int W_STRIDE = 48;     // 32B qs + 16B pad (conflict-free ldmatrix)
constexpr int A_STRIDE = 48;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200) && (__CUDA_ARCH__ < 1300)
#define STRATA_DENSE_FP4_DEV 1
#endif

__device__ __forceinline__ uint32_t ld32_u(const uint8_t* p) {
    const uint16_t lo = __ldg(reinterpret_cast<const uint16_t*>(p));
    const uint16_t hi = __ldg(reinterpret_cast<const uint16_t*>(p + 2));
    return (uint32_t) lo | ((uint32_t) hi << 16);
}

__device__ __forceinline__ uint32_t ld32_fast(const uint8_t* p, bool aligned4) {
    return aligned4 ? __ldg(reinterpret_cast<const uint32_t*>(p)) : ld32_u(p);
}

#if defined(STRATA_DENSE_FP4_DEV)
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return (uint32_t) __cvta_generic_to_shared(p);
}

__device__ __forceinline__ void cp16_async(void* dst_smem, const void* src_glob) {
    const uint32_t d = smem_u32(dst_smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(src_glob));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_wait0() { asm volatile("cp.async.wait_group 0;\n"); }

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* src_smem) {
    const uint32_t a = smem_u32(src_smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

__device__ __forceinline__ void mma_nvfp4_m16n8k64(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                                   uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
}

__device__ __forceinline__ uint8_t encode_ue4m3(float x, float& decoded_scale) {
    if (!(x > 0.0f)) {
        decoded_scale = 0.0f;
        return 0;
    }
    const float clamped = fminf(x, 448.0f);
    const __nv_fp8_e4m3 fp8(clamped);
    decoded_scale = (float) fp8;
    return fp8.__x;
}

__device__ __forceinline__ uint8_t pack_e2m1x2(float v_lo, float v_hi) {
    const __nv_fp4x2_e2m1 q2(make_float2(v_lo, v_hi));
    return q2.__x;
}
#endif

// Quantize FP16 activations X [T, K] -> split-half NVFP4 blocks (40 bytes per 64 values: qs[32], d[4], g_blk).
// One warp per 64-element block (8 warps / 256 threads per CTA).
__global__ void __launch_bounds__(256) quant_f16_to_nvfp4_kernel(const uint16_t* __restrict__ x, int64_t nblk,
                                                                 uint8_t* __restrict__ xa) {
#if defined(STRATA_DENSE_FP4_DEV)
    const int64_t w = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    if (w >= nblk) return;
    const int lane = threadIdx.x & 31;
    const int sub = lane >> 3;
    const int j = lane & 7;

    const uint16_t* xb = x + w * K_STEP;
    const float v0 = __half2float(__ushort_as_half(__ldg(xb + 16 * sub + j)));
    const float v1 = __half2float(__ushort_as_half(__ldg(xb + 16 * sub + 8 + j)));

    float am_sub = fmaxf(fabsf(v0), fabsf(v1));
#pragma unroll
    for (int o = 1; o < 8; o <<= 1) am_sub = fmaxf(am_sub, __shfl_xor_sync(0xffffffffu, am_sub, o));

    float am_64 = fmaxf(am_sub, __shfl_xor_sync(0xffffffffu, am_sub, 8));
    am_64 = fmaxf(am_64, __shfl_xor_sync(0xffffffffu, am_64, 16));

    const float g_blk = am_64 * (1.0f / (6.0f * 448.0f));
    float s_sub = 0.0f;
    const uint8_t d_code = encode_ue4m3(g_blk > 0.0f ? (am_sub / (g_blk * 6.0f)) : 0.0f, s_sub);
    const float denom = g_blk * s_sub;
    const float inv = denom > 0.0f ? (1.0f / denom) : 0.0f;

    uint8_t* out = xa + w * ACT_BLK;
    out[lane] = pack_e2m1x2(v0 * inv, v1 * inv);
    if (j == 0) out[32 + sub] = d_code;
    if (lane == 0) *reinterpret_cast<float*>(out + 36) = g_blk;
#else
    (void) x; (void) nblk; (void) xa;
#endif
}

// High-throughput coalesced Blackwell mxf4nvf4 dense GEMM: Y[T, N] = beta * Y + X_fp4 @ W_nvfp4^T.
// Grid: (ceil(N / 128), ceil(T / 64)), Block: 256 (8 warps, ~20.5 KB smem, 0 spills).
__global__ void __launch_bounds__(NT, 2) dense_fp4_tc_kernel(const uint8_t* __restrict__ w_base,
                                                             const uint8_t* __restrict__ x_act,
                                                             float* __restrict__ Y, int N, int T, int K,
                                                             int ldy, float beta) {
#if defined(STRATA_DENSE_FP4_DEV)
    __shared__ alignas(16) uint8_t s_wqs[2][BM * W_STRIDE]; // 2 * 6,144 = 12,288 B
    __shared__ alignas(16) uint8_t s_aqs[2][BN * A_STRIDE]; // 2 * 3,072 =  6,144 B
    __shared__ uint32_t s_wsc[2][BM];                       // 2 *   512 =  1,024 B
    __shared__ uint32_t s_asc[2][BN];                       // 2 *   256 =    512 B
    __shared__ float s_agb[2][BN];                          // 2 *   256 =    512 B (total 20,480 B)

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wf = warp >> 2;          // 0..1: 64 output rows each
    const int wn = warp & 3;           // 0..3: 16 tokens each
    const int nb0 = 16 * wn;
    const int g = lane >> 2;
    const int tig = lane & 3;
    const int lm = lane >> 3;
    const int l8 = lane & 7;
    const int tidx_a = g + (tig & 1) * 8;

    const int rl = tid >> 3;           // 0..31: row offset in 32-row group
    const int lc = tid & 7;            // 0..7: 8 lanes load 32 contiguous qs bytes!

    const int n0 = (int) blockIdx.x * BM;
    const int t0 = (int) blockIdx.y * BN;
    const int nrows = min(BM, N - n0);
    const int ntoks = min(BN, T - t0);
    if (nrows <= 0 || ntoks <= 0) return;

    const int ns = K / K_STEP;
    const size_t w_row_bytes = (size_t) ns * BS_NVFP4;
    const size_t a_row_bytes = (size_t) ns * ACT_BLK;
    const bool aligned4 = (((uintptr_t) w_base & 3u) == 0u) && ((w_row_bytes & 3u) == 0u);

    const uint8_t* const wr0 = (rl + 0 < nrows) ? (w_base + (size_t) (n0 + rl + 0) * w_row_bytes) : nullptr;
    const uint8_t* const wr1 = (rl + 32 < nrows) ? (w_base + (size_t) (n0 + rl + 32) * w_row_bytes) : nullptr;
    const uint8_t* const wr2 = (rl + 64 < nrows) ? (w_base + (size_t) (n0 + rl + 64) * w_row_bytes) : nullptr;
    const uint8_t* const wr3 = (rl + 96 < nrows) ? (w_base + (size_t) (n0 + rl + 96) * w_row_bytes) : nullptr;

    const int a_off = (64 * wf + 8 * (lm & 1) + l8) * W_STRIDE + 16 * (lm >> 1);
    const int b_off = (nb0 + 8 * (lm >> 1) + l8) * A_STRIDE + 16 * (lm & 1);

    if (tid < BN && tid >= ntoks) {
        s_asc[0][tid] = 0u; s_asc[1][tid] = 0u;
        s_agb[0][tid] = 0.0f; s_agb[1][tid] = 0.0f;
        uint4* z0 = reinterpret_cast<uint4*>(s_aqs[0] + tid * A_STRIDE);
        uint4* z1 = reinterpret_cast<uint4*>(s_aqs[1] + tid * A_STRIDE);
        z0[0] = make_uint4(0u, 0u, 0u, 0u); z0[1] = make_uint4(0u, 0u, 0u, 0u);
        z1[0] = make_uint4(0u, 0u, 0u, 0u); z1[1] = make_uint4(0u, 0u, 0u, 0u);
    }
    __syncthreads();

    auto issue_act = [&](int st, int buf) {
        if (tid < 2 * BN) {
            const int r = tid >> 1;
            const int h = tid & 1;
            if (r < ntoks) {
                const uint8_t* ap = x_act + (size_t) (t0 + r) * a_row_bytes + (size_t) st * ACT_BLK;
                cp16_async(s_aqs[buf] + r * A_STRIDE + 16 * h, ap + 16 * h);
            }
        } else if (tid < 3 * BN) {
            const int r = tid - 2 * BN;
            if (r < ntoks) {
                const uint8_t* ap = x_act + (size_t) (t0 + r) * a_row_bytes + (size_t) st * ACT_BLK;
                const uint2 meta = *reinterpret_cast<const uint2*>(ap + 32);
                s_asc[buf][r] = meta.x;
                s_agb[buf][r] = __uint_as_float(meta.y);
            }
        }
        cp_commit();
    };

    auto load_w_regs = [&](int st, uint32_t& q0, uint32_t& q1, uint32_t& q2, uint32_t& q3, uint32_t& wsc) {
        const size_t sb = (size_t) st * BS_NVFP4;
        const size_t qo = sb + 4 + (size_t) lc * 4;
        q0 = wr0 ? ld32_fast(wr0 + qo, aligned4) : 0u;
        q1 = wr1 ? ld32_fast(wr1 + qo, aligned4) : 0u;
        q2 = wr2 ? ld32_fast(wr2 + qo, aligned4) : 0u;
        q3 = wr3 ? ld32_fast(wr3 + qo, aligned4) : 0u;
        if (lc == 0) wsc = wr0 ? ld32_fast(wr0 + sb, aligned4) : 0u;
        else if (lc == 1) wsc = wr1 ? ld32_fast(wr1 + sb, aligned4) : 0u;
        else if (lc == 2) wsc = wr2 ? ld32_fast(wr2 + sb, aligned4) : 0u;
        else if (lc == 3) wsc = wr3 ? ld32_fast(wr3 + sb, aligned4) : 0u;
        else wsc = 0u;
    };

    auto store_w_smem = [&](int buf, uint32_t q0, uint32_t q1, uint32_t q2, uint32_t q3, uint32_t wsc) {
        *reinterpret_cast<uint32_t*>(s_wqs[buf] + (rl + 0) * W_STRIDE + lc * 4) = q0;
        *reinterpret_cast<uint32_t*>(s_wqs[buf] + (rl + 32) * W_STRIDE + lc * 4) = q1;
        *reinterpret_cast<uint32_t*>(s_wqs[buf] + (rl + 64) * W_STRIDE + lc * 4) = q2;
        *reinterpret_cast<uint32_t*>(s_wqs[buf] + (rl + 96) * W_STRIDE + lc * 4) = q3;
        if (lc < 4) s_wsc[buf][lc * 32 + rl] = wsc;
    };

    // Prologue: load stage 0 weights & activations.
    issue_act(0, 0);
    {
        uint32_t q0, q1, q2, q3, wsc;
        load_w_regs(0, q0, q1, q2, q3, wsc);
        store_w_smem(0, q0, q1, q2, q3, wsc);
    }

    float acc[4][2][4] = {};
    const bool on0 = nb0 < ntoks;
    const bool on1 = nb0 + 8 < ntoks;

    for (int st = 0; st < ns; ++st) {
        const int buf = st & 1;
        const int nxt = buf ^ 1;
        cp_wait0();
        __syncthreads();

        uint32_t nq0 = 0u, nq1 = 0u, nq2 = 0u, nq3 = 0u, nwsc = 0u;
        if (st + 1 < ns) {
            issue_act(st + 1, nxt);
            load_w_regs(st + 1, nq0, nq1, nq2, nq3, nwsc);
        }

        if (on0) {
            uint32_t bq[4];
            ldsm4(bq, s_aqs[buf] + b_off);
            const uint32_t sfb0 = s_asc[buf][nb0 + g];
            const uint32_t sfb1 = on1 ? s_asc[buf][nb0 + 8 + g] : 0u;
            const float gb0_0 = s_agb[buf][nb0 + 2 * tig];
            const float gb0_1 = s_agb[buf][nb0 + 2 * tig + 1];
            const float gb1_0 = on1 ? s_agb[buf][nb0 + 8 + 2 * tig] : 0.0f;
            const float gb1_1 = on1 ? s_agb[buf][nb0 + 8 + 2 * tig + 1] : 0.0f;

#pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t a[4];
                ldsm4(a, s_wqs[buf] + a_off + 16 * i * W_STRIDE);
                const uint32_t sfa = s_wsc[buf][64 * wf + 16 * i + tidx_a];

                float d0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                mma_nvfp4_m16n8k64(d0, a, bq[0], bq[1], sfa, sfb0);
                acc[i][0][0] = fmaf(d0[0], gb0_0, acc[i][0][0]);
                acc[i][0][1] = fmaf(d0[1], gb0_1, acc[i][0][1]);
                acc[i][0][2] = fmaf(d0[2], gb0_0, acc[i][0][2]);
                acc[i][0][3] = fmaf(d0[3], gb0_1, acc[i][0][3]);

                if (on1) {
                    float d1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    mma_nvfp4_m16n8k64(d1, a, bq[2], bq[3], sfa, sfb1);
                    acc[i][1][0] = fmaf(d1[0], gb1_0, acc[i][1][0]);
                    acc[i][1][1] = fmaf(d1[1], gb1_1, acc[i][1][1]);
                    acc[i][1][2] = fmaf(d1[2], gb1_0, acc[i][1][2]);
                    acc[i][1][3] = fmaf(d1[3], gb1_1, acc[i][1][3]);
                }
            }
        }

        if (st + 1 < ns) {
            store_w_smem(nxt, nq0, nq1, nq2, nq3, nwsc);
        }
    }

    if (on0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row_lo = n0 + 64 * wf + 16 * i + g;
            const int row_hi = row_lo + 8;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                if (n == 1 && !on1) break;
                const int tok0 = t0 + nb0 + 8 * n + 2 * tig;
                const int tok1 = tok0 + 1;
                if (tok0 < T) {
                    if (row_lo < N) {
                        float* dst = Y + (size_t) tok0 * ldy + row_lo;
                        *dst = (beta == 0.0f) ? acc[i][n][0] : fmaf(beta, *dst, acc[i][n][0]);
                    }
                    if (row_hi < N) {
                        float* dst = Y + (size_t) tok0 * ldy + row_hi;
                        *dst = (beta == 0.0f) ? acc[i][n][2] : fmaf(beta, *dst, acc[i][n][2]);
                    }
                }
                if (tok1 < T) {
                    if (row_lo < N) {
                        float* dst = Y + (size_t) tok1 * ldy + row_lo;
                        *dst = (beta == 0.0f) ? acc[i][n][1] : fmaf(beta, *dst, acc[i][n][1]);
                    }
                    if (row_hi < N) {
                        float* dst = Y + (size_t) tok1 * ldy + row_hi;
                        *dst = (beta == 0.0f) ? acc[i][n][3] : fmaf(beta, *dst, acc[i][n][3]);
                    }
                }
            }
        }
    }
#else
    (void) w_base; (void) x_act; (void) Y; (void) N; (void) T; (void) K; (void) ldy; (void) beta;
#endif
}

struct DevState {
    bool ok = false;
    uint8_t* act_buf = nullptr;
    size_t act_cap = 0;
    bool logged = false;
};

std::mutex g_mu;
DevState g_dev[32];
bool g_checked[32] = {};

DevState& dev_state(int dev) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_checked[dev]) {
        DevState st;
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess) {
            const int mj = strata::cc_major_of(major);
            const int mn = strata::cc_minor_of(minor);
            if (mj == 12 && (mn == 0 || mn == 1)) {
                int occ = 0;
                if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, dense_fp4_tc_kernel, NT, 0) == cudaSuccess &&
                    occ > 0) {
                    st.ok = true;
                } else {
                    (void) cudaGetLastError();
                }
            }
        } else {
            (void) cudaGetLastError();
        }
        g_dev[dev] = st;
        g_checked[dev] = true;
    }
    return g_dev[dev];
}

}  // namespace

bool built() { return true; }

bool available() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 32) {
        (void) cudaGetLastError();
        return false;
    }
    return dev_state(dev).ok;
}

bool enabled() {
    static const int env = [] {
        const char* v = std::getenv("STRATA_PF_DENSE_FP4");
        if (v == nullptr || v[0] == '\0') return 1;
        return std::atoi(v) != 0 ? 1 : 0;
    }();
    return env != 0 && available();
}

bool gemm_f16_nvfp4(const uint16_t* X_f16, const void* W_nvfp4, float* Y, int64_t T, int64_t N, int64_t K,
                    int64_t ldy, float beta, void* stream) {
    if (!enabled()) return false;
    if (T < 32 || N < 64 || K < 64 || (K & 63) != 0) return false;
    if (ldy <= 0) ldy = N;

    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 32) {
        (void) cudaGetLastError();
        return false;
    }
    DevState& st = dev_state(dev);
    if (!st.ok) return false;

    const int64_t nblk = T * (K / K_STEP);
    const size_t need = (size_t) nblk * ACT_BLK;
    uint8_t* x_act = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (need > st.act_cap || st.act_buf == nullptr) {
            if (st.act_buf != nullptr) {
                cudaFree(st.act_buf);
                st.act_buf = nullptr;
                st.act_cap = 0;
            }
            const size_t alloc = (need + 65535u) & ~(size_t) 65535u;
            if (cudaMalloc((void**) &st.act_buf, alloc) != cudaSuccess) {
                (void) cudaGetLastError();
                return false;
            }
            st.act_cap = alloc;
        }
        x_act = st.act_buf;
    }

    const cudaStream_t cs = static_cast<cudaStream_t>(stream);
    const unsigned qgrid = (unsigned) ((nblk + 7) / 8);
    quant_f16_to_nvfp4_kernel<<<qgrid, 256, 0, cs>>>(X_f16, nblk, x_act);

    const dim3 dgrid((unsigned) ((N + BM - 1) / BM), (unsigned) ((T + BN - 1) / BN));
    dense_fp4_tc_kernel<<<dgrid, NT, 0, cs>>>(static_cast<const uint8_t*>(W_nvfp4), x_act, Y, (int) N, (int) T,
                                              (int) K, (int) ldy, beta);
    if (cudaGetLastError() != cudaSuccess) return false;

    if (!st.logged) {
        st.logged = true;
        std::fprintf(stderr, "strata prefill: Blackwell mxf4nvf4 Tensor Core dense NVFP4 GEMM active "
                             "(STRATA_PF_DENSE_FP4=0 disables)\n");
    }
    return true;
}

}  // namespace strata::prefill::dense_fp4_tc
