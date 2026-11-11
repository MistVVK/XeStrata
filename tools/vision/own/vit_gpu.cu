// tools/vision/own/vit_gpu.cu - see vit_gpu.h.  One source for CUDA and HIP (STRATA_VIT_HIP maps the few runtime / BLAS names).
#include "vit_gpu.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(STRATA_VIT_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#if defined(STRATA_VIT_HIPBLASLT)
// hipBLASLt for the matrix products (STRATA_VIT_LT / STRATA_VIT_LT_ATT, off by default): on gfx1151 the package's rocBLAS carries only its generic
// fallback kernels, and the products took 1.7 to 3.0 times mtmd-HIP's whole picture (RESULTS-OWN-3V.md 12.14, section 14)
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <map>
#include <tuple>
#endif
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaSetDevice hipSetDevice
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemset hipMemset
#define cudaGetLastError hipGetLastError
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaMemGetInfo hipMemGetInfo
#define cudaDataType_t hipDataType
#define cublasHandle_t hipblasHandle_t
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasStatus_t hipblasStatus_t
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUDA_R_32F HIP_R_32F
#define CUDA_R_16F HIP_R_16F
#define CUDA_R_16BF HIP_R_16BF
#define VIT_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define VIT_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define cublasGemmEx hipblasGemmEx
#define cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
#define VIT_SHFL_XOR(v, m) __shfl_xor((v), (m))
#else
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define VIT_COMPUTE_32F CUBLAS_COMPUTE_32F
#define VIT_GEMM_DEFAULT CUBLAS_GEMM_DEFAULT
#define VIT_SHFL_XOR(v, m) __shfl_xor_sync(0xffffffffu, (v), (m))
#endif

namespace strata_vit {

namespace {

// the 16-bit kind a GEMM's input takes (its weight's).  K_BH: a BF16 matrix held as F16 (every BF16 value of the weights' and activations' range is an F16 value;
// the activations are rounded to BF16 first, so the products see the BF16 values), for cards whose BLAS has no fast BF16 kernel (Pascal: STRATA_VIT_BF16_AS_F16)
enum Kind { K_F32 = 0, K_F16 = 1, K_BF16 = 2, K_BH = 3 };

struct DevW {            // a weight matrix [rows = out][cols = in] on the device
    void* p = nullptr;
    Kind kind = K_BF16;
    int rows = 0, cols = 0;
};

struct DevLayer {
    float *ln1_w = nullptr, *ln1_b = nullptr, *ln2_w = nullptr, *ln2_b = nullptr, *qkv_b = nullptr, *o_b = nullptr, *up_b = nullptr, *down_b = nullptr;
    DevW qkv, o, up, down;
};

__device__ __forceinline__ uint16_t f_to_bf16(float f) {          // round to nearest even
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40);   // NaN stays NaN
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
__device__ __forceinline__ uint16_t f_to_f16bits(float f) { return __half_as_ushort(__float2half_rn(f)); }

__device__ __forceinline__ void store_kind(void* out, size_t i, float v, int kind) {
    if (kind == K_F32) ((float*) out)[i] = v;
    else if (kind == K_F16) ((uint16_t*) out)[i] = f_to_f16bits(v);
    else if (kind == K_BH) ((uint16_t*) out)[i] = f_to_f16bits(__uint_as_float((uint32_t) f_to_bf16(v) << 16));
    else ((uint16_t*) out)[i] = f_to_bf16(v);
}


// block-wide sum of one float per thread (blockDim.x a multiple of the warp size, at most 1024)
__device__ float block_sum(float v) {
    __shared__ float red[32];
    const int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
    for (int o = warpSize / 2; o > 0; o >>= 1) v += VIT_SHFL_XOR(v, o);
    __syncthreads();
    if (lane == 0) red[wid] = v;
    __syncthreads();
    const int nw = (blockDim.x + warpSize - 1) / warpSize;
    v = lane < nw ? red[lane] : 0.0f;
    for (int o = warpSize / 2; o > 0; o >>= 1) v += VIT_SHFL_XOR(v, o);
    return v;
}


// x[t] += patch bias + the position table resized (bilinear, align corners: ggml's maths, in float) to the token's grid position
__global__ void k_patch_post(float* __restrict__ x, const float* __restrict__ bias, const float* __restrict__ pos, int G, int pw, int ph, int nbx, int D) {
    const int t = blockIdx.x, b = t / 4, r = t % 4;
    const int ty = 2 * (b / nbx) + r / 2, tx = 2 * (b % nbx) + r % 2;
    float sfx = (float) pw / G, sfy = (float) ph / G;
    if (pw > 1 && G > 1) sfx = (float) (pw - 1) / (G - 1);
    if (ph > 1 && G > 1) sfy = (float) (ph - 1) / (G - 1);
    const float fy = (float) ty / sfy, fx = (float) tx / sfx;
    int y0 = (int) floorf(fy), x0 = (int) floorf(fx), y1 = y0 + 1, x1 = x0 + 1;
    y0 = max(0, min(y0, G - 1)); y1 = max(0, min(y1, G - 1));
    x0 = max(0, min(x0, G - 1)); x1 = max(0, min(x1, G - 1));
    const float dy = fmaxf(0.0f, fminf(fy - (float) y0, 1.0f)), dx = fmaxf(0.0f, fminf(fx - (float) x0, 1.0f));
    const float* a = pos + ((size_t) y0 * G + x0) * D;
    const float* bb = pos + ((size_t) y0 * G + x1) * D;
    const float* c = pos + ((size_t) y1 * G + x0) * D;
    const float* d = pos + ((size_t) y1 * G + x1) * D;
    float* xr = x + (size_t) t * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        xr[i] += bias[i] + (a[i] * (1 - dx) * (1 - dy) + bb[i] * dx * (1 - dy) + c[i] * (1 - dx) * dy + d[i] * dx * dy);
}

// out = LayerNorm(x) * w + b, into `kind`
__global__ void k_layernorm(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b, float eps, int D, void* out, int kind) {
    const float* xr = x + (size_t) blockIdx.x * D;
    float s = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) s += xr[i];
    const float mean = block_sum(s) / (float) D;
    float v = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) { const float d = xr[i] - mean; v += d * d; }
    const float scale = rsqrtf(block_sum(v) / (float) D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) store_kind(out, (size_t) blockIdx.x * D + i, (xr[i] - mean) * scale * w[i] + b[i], kind);
}

// y [rows][3D] (the fused qkv product of the tokens row0 ..) + bias -> q, k (rotary applied), v, each [N][D] in F16 (as ggml's flash attention holds
// them).  Pair j of a head is (j, j + hd/2); j < hd/4 turns with the token's row, the others with its column.
__global__ void k_qkv_rope(const float* __restrict__ y, const float* __restrict__ bias, int row0, int nbx, int heads, int hd, float theta_scale, float qscale,
                           uint16_t* __restrict__ q, uint16_t* __restrict__ k, uint16_t* __restrict__ v) {
    const int t = row0 + blockIdx.x, D = heads * hd, half = hd / 2, quarter = hd / 4;
    const int b = t / 4, r = t % 4;
    const int ty = 2 * (b / nbx) + r / 2, tx = 2 * (b % nbx) + r % 2;
    const float* yr = y + (size_t) blockIdx.x * 3 * D;
    for (int i = threadIdx.x; i < 2 * heads * half; i += blockDim.x) {
        const int part = i / (heads * half), rem = i % (heads * half), h = rem / half, j = rem % half;
        const int base = part * D + h * hd;
        const float a = yr[base + j] + bias[base + j], c = yr[base + j + half] + bias[base + j + half];
        const float freq = powf(theta_scale, (float) (j % quarter));
        float sn, cs;
        sincosf((float) (j < quarter ? ty : tx) * freq, &sn, &cs);
        uint16_t* dst = (part == 0 ? q : k) + (size_t) t * D + h * hd;
        const float sc = part == 0 ? qscale : 1.0f;        // the attention's 1/sqrt(head size) goes into q: the scores are then smaller, and an F16 score keeps more digits
        dst[j] = f_to_f16bits((a * cs - c * sn) * sc);
        dst[j + half] = f_to_f16bits((a * sn + c * cs) * sc);
    }
    for (int e = threadIdx.x; e < D; e += blockDim.x) v[(size_t) t * D + e] = f_to_f16bits(yr[2 * D + e] + bias[2 * D + e]);
}

// In place: a row of n F16 scores (the scale is already in q) becomes the row of probabilities, F16.  One block (256 threads) per row; the row is held in
// registers (at most 256 * kSoftmaxPer = 16,384 scores), read once and written once.
constexpr int kSoftmaxThreads = 256, kSoftmaxPer = 64;
// T threads and PER scores per thread (T * PER >= n): a short row (a small picture) gets a small block and a short loop, as the long loop with its
// predicated steps cost a 93-token picture 9 of its 26 ms
template <int T, int PER>
__global__ void __launch_bounds__(T) k_softmax_rows(uint16_t* __restrict__ S, int n, int ld) {
    uint16_t* s = S + (size_t) blockIdx.x * ld;
    __shared__ float red[32];
    const int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize, nw = (blockDim.x + warpSize - 1) / warpSize;
    float v[PER];
    float m = -FLT_MAX;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        const int j = threadIdx.x + k * T;
        v[k] = j < n ? __half2float(__ushort_as_half(s[j])) : -FLT_MAX;
        m = fmaxf(m, v[k]);
    }
    for (int o = warpSize / 2; o > 0; o >>= 1) m = fmaxf(m, VIT_SHFL_XOR(m, o));
    if (lane == 0) red[wid] = m;
    __syncthreads();
    m = lane < nw ? red[lane] : -FLT_MAX;
    for (int o = warpSize / 2; o > 0; o >>= 1) m = fmaxf(m, VIT_SHFL_XOR(m, o));
    __syncthreads();
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        const int j = threadIdx.x + k * T;
        v[k] = j < n ? expf(v[k] - m) : 0.0f;
        sum += v[k];
    }
    for (int o = warpSize / 2; o > 0; o >>= 1) sum += VIT_SHFL_XOR(sum, o);
    if (lane == 0) red[wid] = sum;
    __syncthreads();
    sum = lane < nw ? red[lane] : 0.0f;
    for (int o = warpSize / 2; o > 0; o >>= 1) sum += VIT_SHFL_XOR(sum, o);
    const float inv = 1.0f / sum;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        const int j = threadIdx.x + k * T;
        if (j < n) s[j] = f_to_f16bits(v[k] * inv);
    }
}

// launches the short-row kernel with the smallest block and loop that hold n scores (n <= 256 * 64)
static void softmax_rows(uint16_t* S, int n, int ld, int rows) {
    if (n <= 128) k_softmax_rows<64, 2><<<rows, 64>>>(S, n, ld);
    else if (n <= 512) k_softmax_rows<64, 8><<<rows, 64>>>(S, n, ld);
    else if (n <= 1024) k_softmax_rows<128, 8><<<rows, 128>>>(S, n, ld);
    else if (n <= 4096) k_softmax_rows<256, 16><<<rows, 256>>>(S, n, ld);
    else k_softmax_rows<256, 64><<<rows, 256>>>(S, n, ld);
}

// The same for a row too long to hold in registers (more than 16,384 patches: a picture of 4,096 image tokens or more): three passes over the row in memory.
__global__ void __launch_bounds__(kSoftmaxThreads) k_softmax_rows_long(uint16_t* __restrict__ S, int n, int ld) {
    uint16_t* s = S + (size_t) blockIdx.x * ld;
    __shared__ float red[32];
    const int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize, nw = (blockDim.x + warpSize - 1) / warpSize;
    float m = -FLT_MAX;
    for (int j = threadIdx.x; j < n; j += blockDim.x) m = fmaxf(m, __half2float(__ushort_as_half(s[j])));
    for (int o = warpSize / 2; o > 0; o >>= 1) m = fmaxf(m, VIT_SHFL_XOR(m, o));
    if (lane == 0) red[wid] = m;
    __syncthreads();
    m = lane < nw ? red[lane] : -FLT_MAX;
    for (int o = warpSize / 2; o > 0; o >>= 1) m = fmaxf(m, VIT_SHFL_XOR(m, o));
    __syncthreads();
    float sum = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) sum += expf(__half2float(__ushort_as_half(s[j])) - m);
    for (int o = warpSize / 2; o > 0; o >>= 1) sum += VIT_SHFL_XOR(sum, o);
    if (lane == 0) red[wid] = sum;
    __syncthreads();
    sum = lane < nw ? red[lane] : 0.0f;
    for (int o = warpSize / 2; o > 0; o >>= 1) sum += VIT_SHFL_XOR(sum, o);
    const float inv = 1.0f / sum;
    for (int j = threadIdx.x; j < n; j += blockDim.x) s[j] = f_to_f16bits(expf(__half2float(__ushort_as_half(s[j])) - m) * inv);
}

// y[i] rounded to the 16-bit kind and back (STRATA_VIT_REF_ROUND: ggml's GPU matrix product hands back its result in the weight's 16-bit type)
__global__ void k_round(float* __restrict__ y, size_t n, int kind) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const float v = y[i];
        y[i] = kind == K_F16 ? __half2float(__float2half_rn(v)) : __uint_as_float((uint32_t) f_to_bf16(v) << 16);   // K_BF16 and K_BH: BF16
    }
}

// n floats to the 16-bit kind
__global__ void k_cast(const float* __restrict__ in, size_t n, void* out, int kind) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) store_kind(out, i, in[i], kind);
}

// x[r] += y[r] + bias
__global__ void k_add_bias_res(float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ bias, int D) {
    const size_t base = (size_t) blockIdx.x * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) x[base + i] += y[base + i] + bias[i];
}

// out = gelu_tanh(y + bias) into `kind`
__global__ void k_bias_gelu_cast(const float* __restrict__ y, const float* __restrict__ bias, int n, void* out, int kind) {
    const size_t base = (size_t) blockIdx.x * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float x = y[base + i] + bias[i];
        const float g = 0.5f * x * (1.0f + tanhf(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x)));
        store_kind(out, base + i, g, kind);
    }
}

// y[r] += bias
__global__ void k_add_bias(float* __restrict__ y, const float* __restrict__ bias, int n) {
    const size_t base = (size_t) blockIdx.x * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[base + i] += bias[i];
}

// ---- host side ----
uint16_t host_f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

uint16_t host_f32_to_f16(float f) {
    const __half h = __float2half_rn(f);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}

}  // namespace

struct VitGpuImpl {
    VitGeometry g;
    int device = 0;
    cublasHandle_t blas = nullptr;
    std::vector<void*> allocs;      // every device allocation (weights and work buffers), freed in the destructor
    size_t bytes = 0, work_bytes = 0;
    std::vector<void*> work;        // the work buffers (a prefix of allocs is weights, the rest are these)
    size_t n_weight_allocs = 0;

    float *patch_w = nullptr, *patch_b = nullptr, *pos = nullptr, *post_w = nullptr, *post_b = nullptr, *mm0_b = nullptr, *mm2_b = nullptr;
    DevW mm0, mm2;
    std::vector<DevLayer> layers;

    int cap_patches = 0;
    // Work buffers, sized for cap_patches patches N (D = width).  The GPU memory is what a reader costs next to the engine, so the matrix products run in
    // chunks of kRows tokens: only the residual stream, the attention context and the 16-bit q, k, v and GEMM input are N rows long.
    float *d_x = nullptr, *d_ctx = nullptr, *d_y = nullptr, *d_out = nullptr;   // d_x: residual [N][D]; d_ctx: attention context [N][D] (also the patch matrix, and d_out the
                                                                                 // projector's output at the end); d_y: one chunk's product [kRows][max(3D, ffn, merger)]
    uint16_t *d_a = nullptr, *d_f16 = nullptr;                    // a chunk's GEMM input in its 16-bit kind [kRows][D]; the activated FFN / merger hidden of a chunk [kRows][max(ffn, merger)]
    uint16_t *d_q16 = nullptr, *d_k16 = nullptr, *d_v16 = nullptr;   // q (scaled by 1/sqrt(head size)), k, v [N][D], F16
    uint16_t* d_S = nullptr;       // the scores of a chunk of queries for every head, F16; the softmax turns them into the probabilities in place
    size_t sp_elems = 0;
    int kRows = 512;      // rows per chunk of the matrix products (STRATA_VIT_ROWS overrides: a diagnostic)
    size_t kScoreElems = (size_t) 8 << 20;     // scores per attention chunk, 8M: 16 MB of F16 (STRATA_VIT_SCORES, in M elements, overrides)
    size_t kScoreElemsLarge = (size_t) 512 << 20;   // the chunk above 16,384 patches: 512M scores, 1 GB of F16 (see size_buffers)
    bool scores_env = false;                   // STRATA_VIT_SCORES was given: one size for every picture
    // Above 16,384 patches a score row is padded to a multiple of 64 elements and the queries of an attention chunk are a multiple of 64: with the
    // picture's own row length (limit_7000x5000.png, 65,232 patches: 514 queries per chunk, rows not 128-byte aligned) the A4000 took 108.1 s against
    // 80.3 s aligned, while 65,536 patches took 80.9 s (RESULTS-OWN-3V.md 16.2).  Up to 16,384 patches: 8 and no rounding, as before.
    // STRATA_VIT_NPAD / STRATA_VIT_QALIGN (diagnostics) set both for every picture.
    int n_pad = 0, q_align = 0;      // 0: the rule above
    int pad_for(size_t n) const { return n_pad ? n_pad : n > 16384 ? 64 : 8; }
    int align_for(size_t n) const { return q_align ? q_align : n > 16384 ? 64 : 1; }
    size_t padded(size_t n) const { const size_t p = (size_t) pad_for(n); return (n + p - 1) / p * p; }
    // BF16 matrices held as F16 (K_BH): on by default on Pascal (compute capability 6.x), where cuBLAS has no fast BF16 kernel (a P100 spent 645 ms of a
    // 1,024-token picture in magma_sgemmEx for BF16 against 497 ms with F16 inputs; the 12-token picture 44 ms against 15; RESULTS-OWN-3V.md 16.1).
    // STRATA_VIT_BF16_AS_F16=0/1 (a diagnostic) overrides.
    bool bf16_as_f16 = false;

#if defined(STRATA_VIT_HIPBLASLT)
    // hipBLASLt (HIP builds that found it).  STRATA_VIT_LT=1: the 16-bit weight products; STRATA_VIT_LT_ATT=1: the two batched attention products;
    // STRATA_VIT_LT_SOL=<id>: that solution for the weight products instead of hipBLASLt's first heuristic answer; STRATA_VIT_LT_VERBOSE=1: the solution
    // of every new shape and a summary on stderr.  The solution of a shape is chosen once per process, by id or as the heuristic's first answer that fits
    // the workspace, never by timing, so two processes take the same kernels.  A shape hipBLASLt cannot run goes to hipBLAS as before.
    struct LtPlan {
        bool ok = false;
        int sol = -1;
        hipblasLtMatmulAlgo_t algo{};
        hipblasLtMatmulDesc_t op = nullptr;
        hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    };
    using LtKey = std::tuple<int, int, int, int, int, int, int, int, int, int, int, int, long long, long long, long long>;
    hipblasLtHandle_t lt = nullptr;
    void* lt_ws = nullptr;
    size_t lt_ws_bytes = (size_t) 32 << 20;
    bool lt_weights = false, lt_att = false, lt_verbose = false;
    int lt_sol = -1;
    std::map<LtKey, LtPlan> lt_plans;
    unsigned long long lt_launches = 0, lt_fallbacks = 0;

    static hipblasLtMatrixLayout_t lt_layout(hipDataType t, int rows, int cols, int ld, int batch, long long stride) {
        hipblasLtMatrixLayout_t l = nullptr;
        if (hipblasLtMatrixLayoutCreate(&l, t, rows, cols, ld) != HIPBLAS_STATUS_SUCCESS) return nullptr;
        if (batch > 1) {
            const int32_t bc = batch;
            const int64_t st = stride;
            if (hipblasLtMatrixLayoutSetAttribute(l, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)) != HIPBLAS_STATUS_SUCCESS ||
                hipblasLtMatrixLayoutSetAttribute(l, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &st, sizeof(st)) != HIPBLAS_STATUS_SUCCESS) {
                hipblasLtMatrixLayoutDestroy(l);
                return nullptr;
            }
        }
        return l;
    }
    static void lt_free(LtPlan& p) {
        if (p.op) hipblasLtMatmulDescDestroy(p.op);
        if (p.a) hipblasLtMatrixLayoutDestroy(p.a);
        if (p.b) hipblasLtMatrixLayoutDestroy(p.b);
        if (p.c) hipblasLtMatrixLayoutDestroy(p.c);
        p = LtPlan{};
    }
    // the descriptors and the solution of one call shape (made once, kept for the reader's life)
    LtPlan& lt_plan(bool ta, bool tb, int m, int n, int k, hipDataType at, int lda, long long sa, hipDataType bt, int ldb, long long sb, hipDataType ct, int ldc,
                    long long sc, int batch, int sol) {
        const LtKey key{(int) ta, (int) tb, m, n, k, (int) at, (int) bt, (int) ct, lda, ldb, ldc, batch, sa, sb, sc};
        auto it = lt_plans.find(key);
        if (it != lt_plans.end()) return it->second;
        LtPlan& p = lt_plans[key];
        const hipblasOperation_t oa = ta ? HIPBLAS_OP_T : HIPBLAS_OP_N, ob = tb ? HIPBLAS_OP_T : HIPBLAS_OP_N;
        if (hipblasLtMatmulDescCreate(&p.op, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(p.op, HIPBLASLT_MATMUL_DESC_TRANSA, &oa, sizeof(oa)) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(p.op, HIPBLASLT_MATMUL_DESC_TRANSB, &ob, sizeof(ob)) != HIPBLAS_STATUS_SUCCESS ||
            !(p.a = lt_layout(at, ta ? k : m, ta ? m : k, lda, batch, sa)) || !(p.b = lt_layout(bt, tb ? n : k, tb ? k : n, ldb, batch, sb)) ||
            !(p.c = lt_layout(ct, m, n, ldc, batch, sc))) {
            lt_free(p);
            return p;
        }
        const float alpha = 1.0f, beta = 0.0f;
        size_t need = 0;
        if (sol >= 0) {      // a given solution id
            std::vector<int> ids{sol};
            std::vector<hipblasLtMatmulHeuristicResult_t> r;
            if (hipblaslt_ext::getAlgosFromIndex(lt, ids, r) == HIPBLAS_STATUS_SUCCESS && !r.empty() && r.front().state == HIPBLAS_STATUS_SUCCESS) {
                hipblasLtMatmulAlgo_t algo = r.front().algo;
                if (hipblaslt_ext::matmulIsAlgoSupported(lt, p.op, &alpha, p.a, p.b, &beta, p.c, p.c, algo, need) == HIPBLAS_STATUS_SUCCESS &&
                    need <= lt_ws_bytes) {
                    p.algo = algo;
                    p.ok = true;
                }
            }
        } else {             // hipBLASLt's first heuristic answer that fits the workspace
            hipblasLtMatmulPreference_t pref = nullptr;
            const uint64_t ws = lt_ws_bytes;
            if (hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS &&
                hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)) == HIPBLAS_STATUS_SUCCESS) {
                hipblasLtMatmulHeuristicResult_t r[8];
                int got = 0;
                if (hipblasLtMatmulAlgoGetHeuristic(lt, p.op, p.a, p.b, p.c, p.c, pref, 8, r, &got) == HIPBLAS_STATUS_SUCCESS)
                    for (int i = 0; i < got && !p.ok; ++i)
                        if (r[i].state == HIPBLAS_STATUS_SUCCESS && r[i].workspaceSize <= lt_ws_bytes) {
                            p.algo = r[i].algo;
                            need = r[i].workspaceSize;
                            p.ok = true;
                        }
            }
            if (pref) hipblasLtMatmulPreferenceDestroy(pref);
        }
        if (p.ok) p.sol = hipblaslt_ext::getIndexFromAlgo(p.algo);
        if (lt_verbose)
            std::fprintf(stderr, "vit-lt: %s%s m=%d n=%d k=%d types=%d/%d/%d batch=%d -> %s %d (workspace %zu)\n", ta ? "T" : "N", tb ? "T" : "N", m, n, k, (int) at,
                         (int) bt, (int) ct, batch, p.ok ? "solution" : "hipBLAS, no solution", p.sol, need);
        if (!p.ok) { lt_free(p); p.ok = false; }
        return p;
    }
    // the product through hipBLASLt; false: not run (the caller uses hipBLAS)
    bool lt_mm(bool ta, bool tb, int m, int n, int k, const void* A, cudaDataType_t at, int lda, long long sa, const void* B, cudaDataType_t bt, int ldb,
               long long sb, void* C, cudaDataType_t ct, int ldc, long long sc, int batch, int sol) {
        LtPlan& p = lt_plan(ta, tb, m, n, k, at, lda, sa, bt, ldb, sb, ct, ldc, sc, batch, sol);
        if (!p.ok) { ++lt_fallbacks; return false; }
        const float alpha = 1.0f, beta = 0.0f;
        if (hipblasLtMatmul(lt, p.op, &alpha, A, p.a, B, p.b, &beta, C, p.c, C, p.c, &p.algo, lt_ws, lt_ws_bytes, nullptr) != HIPBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "vit-lt: the hipBLASLt product failed (m=%d n=%d k=%d batch=%d); hipBLAS from now on for this shape\n", m, n, k, batch);
            lt_free(p);
            p.ok = false;
            ++lt_fallbacks;
            return false;
        }
        ++lt_launches;
        return true;
    }
    void lt_setup() {
        auto on = [](const char* name) { const char* e = std::getenv(name); return e && std::atoi(e) != 0; };
        lt_weights = on("STRATA_VIT_LT");
        lt_att = on("STRATA_VIT_LT_ATT");
        lt_verbose = on("STRATA_VIT_LT_VERBOSE");
        if (const char* e = std::getenv("STRATA_VIT_LT_SOL")) lt_sol = std::atoi(e);
        if (!lt_weights && !lt_att) return;
        if (hipblasLtCreate(&lt) != HIPBLAS_STATUS_SUCCESS || cudaMalloc(&lt_ws, lt_ws_bytes) != cudaSuccess) {
            (void) cudaGetLastError();
            std::fprintf(stderr, "vit-lt: hipBLASLt is not usable here; hipBLAS for every product\n");
            if (lt) { hipblasLtDestroy(lt); lt = nullptr; }
            lt_ws = nullptr;
            lt_weights = lt_att = false;
            return;
        }
        std::fprintf(stderr, "vit-lt: hipBLASLt for %s%s%s (solution %s)\n", lt_weights ? "the weight products" : "", lt_weights && lt_att ? " and " : "",
                     lt_att ? "the attention products" : "", lt_sol >= 0 ? std::to_string(lt_sol).c_str() : "heuristic");
    }
    void lt_close() {
        for (auto& kv : lt_plans) lt_free(kv.second);
        lt_plans.clear();
        if (lt_ws) (void) cudaFree(lt_ws);
        if (lt) hipblasLtDestroy(lt);
        if (lt_verbose && (lt_launches || lt_fallbacks))
            std::fprintf(stderr, "vit-lt: %llu hipBLASLt products, %llu by hipBLAS\n", lt_launches, lt_fallbacks);
        lt = nullptr;
        lt_ws = nullptr;
    }
#endif

    // C = alpha * op(A) * op(B), batched, column-major as the BLAS wants it
    bool bgemm(bool ta, bool tb, int m, int n, int k, const void* A, cudaDataType_t at, int lda, long long sa, const void* B, cudaDataType_t bt, int ldb,
               long long sb, void* C, cudaDataType_t ct, int ldc, long long sc, int batch, std::string& err) {
#if defined(STRATA_VIT_HIPBLASLT)
        if (lt_att && lt_mm(ta, tb, m, n, k, A, at, lda, sa, B, bt, ldb, sb, C, ct, ldc, sc, batch, -1)) return true;
#endif
        const float alpha = 1.0f, beta = 0.0f;
        const cublasStatus_t s = cublasGemmStridedBatchedEx(blas, ta ? CUBLAS_OP_T : CUBLAS_OP_N, tb ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha, A, at, lda, sa,
                                                            B, bt, ldb, sb, &beta, C, ct, ldc, sc, batch, VIT_COMPUTE_32F, VIT_GEMM_DEFAULT);
        if (s != CUBLAS_STATUS_SUCCESS) { err = "the batched BLAS product failed (status " + std::to_string((int) s) + ")"; return false; }
        return true;
    }

    ~VitGpuImpl() {
        (void) cudaSetDevice(device);
#if defined(STRATA_VIT_HIPBLASLT)
        lt_close();
#endif
        for (void* p : allocs) (void) cudaFree(p);
        if (blas) cublasDestroy(blas);
    }
    template <class T> bool alloc(T*& p, size_t n, std::string& err, bool weight) {
        void* q = nullptr;
        const cudaError_t e = cudaMalloc(&q, n * sizeof(T));
        if (e != cudaSuccess) { err = std::string("GPU memory: ") + cudaGetErrorString(e); (void) cudaGetLastError(); return false; }
        p = (T*) q;
        allocs.push_back(q);
        bytes += n * sizeof(T);
        if (!weight) work_bytes += n * sizeof(T);
        return true;
    }
    bool upload_f32(float*& dst, const Gguf& gg, const std::string& name, std::string& err) {
        const GgufTensorInfo* t = gg.tensor(name);
        if (!t) { err = "the mmproj file lacks the tensor " + name; return false; }
        std::vector<float> h((size_t) t->nelem());
        if (!gg.read_f32(*t, h.data(), err)) return false;
        if (!alloc(dst, h.size(), err, true)) return false;
        return cudaMemcpy(dst, h.data(), h.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess || (err = "GPU copy failed", false);
    }
    bool upload_matrix(DevW& w, const Gguf& gg, const std::string& name, std::string& err) {
        const GgufTensorInfo* t = gg.tensor(name);
        if (!t || t->ne.size() != 2) { err = "the mmproj file lacks the matrix " + name; return false; }
        w.cols = (int) t->ne[0];
        w.rows = (int) t->ne[1];
        const size_t n = (size_t) t->nelem();
        std::vector<unsigned char> raw;
        if (t->type == GT_BF16 && bf16_as_f16) {                 // BF16 -> F16 (K_BH)
            std::vector<float> f(n);
            if (!gg.read_f32(*t, f.data(), err)) return false;
            raw.resize(n * 2);
            w.kind = K_BH;
            for (size_t i = 0; i < n; ++i) {
                const uint16_t u = host_f32_to_f16(f[i]);
                std::memcpy(&raw[i * 2], &u, 2);
            }
        } else if (t->type == GT_F16 || t->type == GT_BF16) {   // as stored
            raw.resize(n * 2);
            if (!gg.read_raw(*t, raw.data(), err)) return false;
            w.kind = t->type == GT_F16 ? K_F16 : K_BF16;
        } else {                                                 // F32 -> BF16, Q8_0 -> F16
            std::vector<float> f(n);
            if (!gg.read_f32(*t, f.data(), err)) return false;
            raw.resize(n * 2);
            const bool bf = t->type == GT_F32;
            w.kind = bf ? K_BF16 : K_F16;
            for (size_t i = 0; i < n; ++i) {
                const uint16_t u = bf ? host_f32_to_bf16(f[i]) : host_f32_to_f16(f[i]);
                std::memcpy(&raw[i * 2], &u, 2);
            }
        }
        uint16_t* dev = nullptr;
        if (!alloc(dev, n, err, true)) return false;
        w.p = dev;
        return cudaMemcpy(w.p, raw.data(), n * 2, cudaMemcpyHostToDevice) == cudaSuccess || (err = "GPU copy failed", false);
    }
    static cudaDataType_t dtype(Kind k) { return k == K_F32 ? CUDA_R_32F : (k == K_F16 || k == K_BH) ? CUDA_R_16F : CUDA_R_16BF; }

    // y[M][N] (float32) = A[M][K] (kind of w) * w^T
    bool gemm(const DevW& w, const void* A, int M, void* y, cudaDataType_t kind_a, std::string& err) {
        const float alpha = 1.0f, beta = 0.0f;
        bool done = false;
#if defined(STRATA_VIT_HIPBLASLT)
        if (lt_weights && kind_a != CUDA_R_32F)
            done = lt_mm(true, false, w.rows, M, w.cols, w.p, kind_a, w.cols, 0, A, kind_a, w.cols, 0, y, CUDA_R_32F, w.rows, 0, 1, lt_sol);
#endif
        if (!done) {
            const cublasStatus_t s = cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, w.rows, M, w.cols, &alpha, w.p, kind_a, w.cols, A, kind_a, w.cols, &beta, y,
                                                  CUDA_R_32F, w.rows, VIT_COMPUTE_32F, VIT_GEMM_DEFAULT);
            if (s != CUBLAS_STATUS_SUCCESS) { err = "the BLAS matrix product failed (status " + std::to_string((int) s) + ")"; return false; }
        }
        if (round_out && w.kind != K_F32) {   // the result as ggml's GPU path keeps it: 16 bits (its inputs are 16 bits here already)
            const size_t n = (size_t) M * w.rows;
            k_round<<<blocks_for(n), 256>>>((float*) y, n, (int) w.kind);
        }
        return true;
    }
    bool round_out = false;           // STRATA_VIT_REF_ROUND=1 (a diagnostic, off by default): round every matrix product's result to its weight's 16-bit type
    static int blocks_for(size_t n, int per = 256) { return (int) std::min<size_t>((n + per - 1) / per, 65535); }

    bool size_buffers(int patches, std::string& err) {
        if (patches <= cap_patches) return true;
        (void) cudaSetDevice(device);
        // free the old work buffers and take new ones sized for this many patches
        for (size_t i = allocs.size(); i > n_weight_allocs; --i) { (void) cudaFree(allocs[i - 1]); }
        bytes -= work_bytes;
        work_bytes = 0;
        allocs.resize(n_weight_allocs);
        const size_t N = (size_t) patches, D = (size_t) g.width, F = (size_t) std::max(g.ffn, g.merger_hidden);
        const size_t K0 = (size_t) 3 * g.patch * g.patch;
        d_x = d_y = d_ctx = d_out = nullptr;
        d_a = d_f16 = d_q16 = d_k16 = d_v16 = nullptr;
        d_S = nullptr;
        // attention by batched GEMMs: scores S (float32) and probabilities P (F16) of a chunk of queries for every head, at most kScoreElems each
        // (a very large picture gets at least 64 queries per chunk, so its many chunks stay few)
        // a score row is padded to a multiple of 8 elements (the product's leading dimension): with an odd row length the BLAS cannot use its tensor-core
        // kernels (a 93-token picture spent half its time there)
        const size_t Np = padded(N);
        // above 16,384 patches (pictures of more than 4,096 tokens) the chunk is 512M scores (1 GB): with the 16 MB chunk such a picture runs 64 queries per
        // product and the attention is launch-bound (l620, 16,384 tokens: 33.4 s, 1.17 of mtmd-GPU; with 512M 24.8 s, 0.87, at 3,398 MiB against mtmd-GPU's
        // 3,858; RESULTS-OWN-3V.md 12.11).  STRATA_VIT_SCORES sets both sizes.
        const size_t budget = (N > 16384 && !scores_env) ? kScoreElemsLarge : kScoreElems;
        sp_elems = std::min<size_t>((size_t) g.heads * N * Np, std::max<size_t>(budget, (size_t) g.heads * Np * 64));
        const size_t R = std::min<size_t>(N, (size_t) kRows);
        // d_ctx holds, one after the other: the patch matrix, the attention context, and at the end the projector's output rows (d_out)
        const size_t ctx_elems = std::max({N * D, N * K0, (N / 4) * (size_t) g.out_dim});
        bool ok = alloc(d_x, N * D, err, false) && alloc(d_ctx, ctx_elems, err, false) && alloc(d_y, R * std::max<size_t>(3 * D, F), err, false) &&
                  alloc(d_a, R * D, err, false) && alloc(d_f16, R * F, err, false) &&
                  alloc(d_q16, N * D, err, false) && alloc(d_k16, N * D, err, false) && alloc(d_v16, N * D, err, false);
        if (ok && !alloc(d_S, sp_elems, err, false)) {
            // no room for the large chunk (a small card, or the engine beside the reader): the 16 MB chunk instead, slower, never an error
            ok = false;
            if (budget != kScoreElems) {
                err.clear();
                sp_elems = std::min<size_t>((size_t) g.heads * N * Np, std::max<size_t>(kScoreElems, (size_t) g.heads * Np * 64));
                ok = alloc(d_S, sp_elems, err, false);
            }
        }
        d_out = d_ctx;
        if (!ok) { cap_patches = 0; return false; }
        cap_patches = patches;
        return true;
    }
};

VitGpu::VitGpu() : impl_(new VitGpuImpl()) {}
VitGpu::~VitGpu() = default;
size_t VitGpu::device_bytes() const { return impl_->bytes; }

bool VitGpu::available(std::string& why) {
    int n = 0;
    const cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess || n <= 0) { why = e != cudaSuccess ? std::string("no GPU runtime: ") + cudaGetErrorString(e) : "no GPU found"; (void) cudaGetLastError(); return false; }
    return true;
}

bool VitGpu::prefer_mtmd(bool bf16_matrices, int max_tokens, std::string& why) {
#if defined(STRATA_VIT_HIP)
    hipDeviceProp_t p;
    (void) bf16_matrices;
    (void) max_tokens;
    if (hipGetDeviceProperties(&p, 0) != hipSuccess) { (void) hipGetLastError(); return false; }
#ifdef _WIN32
    const bool windows = true;
#else
    const bool windows = false;
#endif
    if (vit_gpu_hip_windows_not_measured(windows, p.gcnArchName)) {
        why = std::string(p.gcnArchName) + " on Windows: the own HIP tower is not measured with the Windows ROCm package yet, mtmd stays";
        return true;
    }
    return false;
#else
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) { (void) cudaGetLastError(); return false; }
    if (vit_gpu_slower_nvidia(p.major, p.minor, bf16_matrices, max_tokens)) {
        why = "compute capability " + std::to_string(p.major) + "." + std::to_string(p.minor) + " (Pascal)" + (bf16_matrices ? "" : " with F16 matrices") +
              ": the own GPU tower is slower than mtmd there";
        return true;
    }
    return false;
#endif
}

bool VitGpu::load(const Gguf& gg, const VitGeometry& g, int device, int max_patches, std::string& err) {
    VitGpuImpl& m = *impl_;
    std::string why;
    if (!vit_geometry_supported(g, why)) { err = "this tower is not supported: " + why; return false; }
    m.g = g;
    m.device = device;
#if !defined(STRATA_VIT_HIP)
    bool pascal = false;
    {   // Pascal (compute capability 6.x): BF16 matrices as F16 (and the larger score chunk below)
        cudaDeviceProp p;
        if (cudaGetDeviceProperties(&p, device) == cudaSuccess && p.major == 6) pascal = m.bf16_as_f16 = true;
        (void) cudaGetLastError();
    }
#endif
    if (const char* e = std::getenv("STRATA_VIT_BF16_AS_F16")) m.bf16_as_f16 = std::atoi(e) != 0;
    if (cudaSetDevice(device) != cudaSuccess) { err = std::string("cannot use GPU ") + std::to_string(device) + ": " + cudaGetErrorString(cudaGetLastError()); return false; }
    if (cublasCreate(&m.blas) != CUBLAS_STATUS_SUCCESS) { err = "cannot create the BLAS handle"; return false; }
#if defined(STRATA_VIT_HIPBLASLT)
    m.lt_setup();
#endif
    // the patch embedding: the two temporal taps summed, kept in float32 (a small matrix)
    {
        const GgufTensorInfo* t0 = gg.tensor("v.patch_embd.weight");
        if (!t0) { err = "the mmproj file lacks v.patch_embd.weight"; return false; }
        std::vector<float> w0((size_t) t0->nelem()), w1;
        if (!gg.read_f32(*t0, w0.data(), err)) return false;
        if (g.patch_taps == 2) {
            const GgufTensorInfo* t1 = gg.tensor("v.patch_embd.weight.1");
            w1.resize(w0.size());
            if (!t1 || !gg.read_f32(*t1, w1.data(), err)) { if (err.empty()) err = "the mmproj file lacks v.patch_embd.weight.1"; return false; }
            for (size_t i = 0; i < w0.size(); ++i) w0[i] += w1[i];
        }
        if (!m.alloc(m.patch_w, w0.size(), err, true)) return false;
        if (cudaMemcpy(m.patch_w, w0.data(), w0.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess) { err = "GPU copy failed"; return false; }
    }
    if (!m.upload_f32(m.patch_b, gg, "v.patch_embd.bias", err) || !m.upload_f32(m.pos, gg, "v.position_embd.weight", err) ||
        !m.upload_f32(m.post_w, gg, "v.post_ln.weight", err) || !m.upload_f32(m.post_b, gg, "v.post_ln.bias", err) ||
        !m.upload_matrix(m.mm0, gg, "mm.0.weight", err) || !m.upload_f32(m.mm0_b, gg, "mm.0.bias", err) ||
        !m.upload_matrix(m.mm2, gg, "mm.2.weight", err) || !m.upload_f32(m.mm2_b, gg, "mm.2.bias", err))
        return false;
    m.layers.assign((size_t) g.layers, DevLayer{});
    for (int il = 0; il < g.layers; ++il) {
        DevLayer& L = m.layers[(size_t) il];
        const std::string p = "v.blk." + std::to_string(il) + ".";
        if (!m.upload_f32(L.ln1_w, gg, p + "ln1.weight", err) || !m.upload_f32(L.ln1_b, gg, p + "ln1.bias", err) ||
            !m.upload_f32(L.ln2_w, gg, p + "ln2.weight", err) || !m.upload_f32(L.ln2_b, gg, p + "ln2.bias", err) ||
            !m.upload_matrix(L.qkv, gg, p + "attn_qkv.weight", err) || !m.upload_f32(L.qkv_b, gg, p + "attn_qkv.bias", err) ||
            !m.upload_matrix(L.o, gg, p + "attn_out.weight", err) || !m.upload_f32(L.o_b, gg, p + "attn_out.bias", err) ||
            !m.upload_matrix(L.up, gg, p + "ffn_up.weight", err) || !m.upload_f32(L.up_b, gg, p + "ffn_up.bias", err) ||
            !m.upload_matrix(L.down, gg, p + "ffn_down.weight", err) || !m.upload_f32(L.down_b, gg, p + "ffn_down.bias", err))
            return false;
    }
    m.n_weight_allocs = m.allocs.size();
#if !defined(STRATA_VIT_HIP)
    // Pascal with BF16 matrices (held as F16): a 32M-score attention chunk (64 MB): the P*V product's tile is only partly used with a head size of 72, and
    // more queries per product made a 1,024-token picture 1,242 -> 1,064 ms on a P100, at 1,362 MiB against mtmd-GPU's 1,364 (RESULTS-OWN-3V.md 16.1).
    // Not with F16 matrices: there mtmd stays faster anyway and the chunk would cost 48 MB over mtmd-GPU's memory.
    if (pascal && !m.layers.empty() && m.layers[0].qkv.kind == K_BH) m.kScoreElems = (size_t) 32 << 20;
#endif
    if (const char* e = std::getenv("STRATA_VIT_REF_ROUND")) m.round_out = std::atoi(e) != 0;
    if (const char* e = std::getenv("STRATA_VIT_ROWS")) { const int v = std::atoi(e); if (v >= 16) m.kRows = v; }
    if (const char* e = std::getenv("STRATA_VIT_SCORES")) { const int v = std::atoi(e); if (v >= 1) { m.kScoreElems = (size_t) v << 20; m.scores_env = true; } }
    // STRATA_VIT_SCORES_LARGE (M elements, a diagnostic): the chunk above 16,384 patches; a size the GPU cannot hold tests the fallback to the default chunk
    if (const char* e = std::getenv("STRATA_VIT_SCORES_LARGE")) { const long v = std::atol(e); if (v >= 1) m.kScoreElemsLarge = (size_t) v << 20; }
    if (const char* e = std::getenv("STRATA_VIT_NPAD")) { const int v = std::atoi(e); if (v >= 8 && v % 8 == 0) m.n_pad = v; }
    if (const char* e = std::getenv("STRATA_VIT_QALIGN")) { const int v = std::atoi(e); if (v >= 1) m.q_align = v; }
    return m.size_buffers(std::max(max_patches, 16), err);
}

bool VitGpu::encode(const PreparedImage& img, std::vector<float>& rows, int& nx, int& ny, std::string& err) {
    VitGpuImpl& m = *impl_;
    const VitGeometry& g = m.g;
    if (m.layers.empty()) { err = "the tower is not loaded"; return false; }
    if (img.w <= 0 || img.h <= 0 || img.w % g.align() != 0 || img.h % g.align() != 0) { err = "the picture size is not a multiple of patch * merge"; return false; }
    if (g.head_dim > 96 || g.head_dim % 4 != 0) { err = "this head size is not handled by the GPU attention kernel"; return false; }
    const int P = g.patch, D = g.width, H = g.heads, HD = g.head_dim, F = g.ffn;
    const int pw = img.w / P, ph = img.h / P, N = pw * ph, nbx = pw / 2;
    nx = pw / 2;
    ny = ph / 2;
    if (cudaSetDevice(m.device) != cudaSuccess) { err = "cannot use the GPU"; return false; }
    if (!m.size_buffers(N, err)) return false;
    auto sync_err = [&](const char* what) {
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { err = std::string(what) + ": " + cudaGetErrorString(e); return false; }
        return true;
    };
    // STRATA_VIT_PROF=1 (a diagnostic, stderr only, off by default): the time of each stage, the GPU synchronised after every stage
    static const bool prof = std::getenv("STRATA_VIT_PROF") != nullptr;
    double pt[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    auto pclock = [] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    double pmark = prof ? pclock() : 0.0;
    auto stage = [&](int i) {
        if (!prof) return;
        (void) cudaDeviceSynchronize();
        const double now = pclock();
        pt[i] += now - pmark;
        pmark = now;
    };
    const int K0 = 3 * P * P;
    {   // the patch matrix [N][K0], channel-major per patch, tokens in 2x2-merge order, built on the host; it lives in the context buffer (free until the first attention)
        std::vector<float> pm((size_t) N * K0);
        const size_t plane = (size_t) img.w * img.h;
        for (int t = 0; t < N; ++t) {
            const int b = t / 4, r = t % 4, py = 2 * (b / nbx) + r / 2, px = 2 * (b % nbx) + r % 2;
            float* a = &pm[(size_t) t * K0];
            for (int c = 0; c < 3; ++c)
                for (int ky = 0; ky < P; ++ky)
                    std::memcpy(a + (size_t) c * P * P + (size_t) ky * P, &img.chw[c * plane + (size_t) (py * P + ky) * img.w + (size_t) px * P], (size_t) P * sizeof(float));
        }
        if (cudaMemcpy(m.d_ctx, pm.data(), pm.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess) { err = "GPU copy failed"; return false; }
    }
    stage(0);
    // patch embedding (float32 GEMM), then bias + position
    DevW pw_w;
    pw_w.p = m.patch_w; pw_w.kind = K_F32; pw_w.rows = D; pw_w.cols = K0;
    if (!m.gemm(pw_w, m.d_ctx, N, m.d_x, CUDA_R_32F, err)) return false;
    k_patch_post<<<N, 256>>>(m.d_x, m.patch_b, m.pos, g.pos_grid, pw, ph, nbx, D);
    if (!sync_err("patch embedding")) return false;
    stage(1);

    const float theta_scale = powf(g.rope_theta, -2.0f / (float) (HD / 2));
    const float kq_scale = 1.0f / sqrtf((float) HD);
    const int R = m.kRows;
    uint16_t* const a16 = m.d_a;
    for (const DevLayer& L : m.layers) {
        for (int r0 = 0; r0 < N; r0 += R) {      // the fused projection in chunks of rows (norm, product); q and k get their rotary, all three go to F16
            const int rows = std::min(R, N - r0);
            k_layernorm<<<rows, 256>>>(m.d_x + (size_t) r0 * D, L.ln1_w, L.ln1_b, g.eps, D, a16, (int) L.qkv.kind);
            if (!m.gemm(L.qkv, a16, rows, m.d_y, VitGpuImpl::dtype(L.qkv.kind), err)) return false;
            k_qkv_rope<<<rows, 256>>>(m.d_y, L.qkv_b, r0, nbx, H, HD, theta_scale, kq_scale, m.d_q16, m.d_k16, m.d_v16);
        }
        stage(2);
        // per chunk of queries: S = K Q^T (one batched product over the heads), softmax, O = V P (a head's matrix is a column slice of the [N][D] arrays:
        // leading dimension D, batch stride HD)
        const int Np = (int) m.padded((size_t) N);      // the score rows' leading dimension (see size_buffers)
        int QC = (int) std::max<size_t>(1, std::min<size_t>((size_t) N, m.sp_elems / ((size_t) H * Np)));
        const int qa = m.align_for((size_t) N);
        if (qa > 1 && QC < N && QC >= qa) QC -= QC % qa;
        for (int q0 = 0; q0 < N; q0 += QC) {
            const int qc = std::min(QC, N - q0);
            if (!m.bgemm(true, false, N, qc, HD, m.d_k16, CUDA_R_16F, D, HD, m.d_q16 + (size_t) q0 * D, CUDA_R_16F, D, HD, m.d_S, CUDA_R_16F, Np,
                         (long long) qc * Np, H, err))
                return false;
            stage(8);
            if (N <= kSoftmaxThreads * kSoftmaxPer) softmax_rows(m.d_S, N, Np, H * qc);
            else k_softmax_rows_long<<<H * qc, kSoftmaxThreads>>>(m.d_S, N, Np);
            stage(9);
            if (!m.bgemm(false, false, HD, qc, N, m.d_v16, CUDA_R_16F, D, HD, m.d_S, CUDA_R_16F, Np, (long long) qc * Np, m.d_ctx + (size_t) q0 * D,
                         CUDA_R_32F, D, HD, H, err))
                return false;
        }
        stage(3);
        for (int r0 = 0; r0 < N; r0 += R) {      // the output projection in chunks: the context rows to the 16-bit input, the product, added to the residual
            const int rows = std::min(R, N - r0);
            k_cast<<<VitGpuImpl::blocks_for((size_t) rows * D), 256>>>(m.d_ctx + (size_t) r0 * D, (size_t) rows * D, a16, (int) L.o.kind);
            if (!m.gemm(L.o, a16, rows, m.d_y, VitGpuImpl::dtype(L.o.kind), err)) return false;
            k_add_bias_res<<<rows, 256>>>(m.d_x + (size_t) r0 * D, m.d_y, L.o_b, D);
        }
        stage(4);
        for (int r0 = 0; r0 < N; r0 += R) {      // the feed-forward block in chunks: norm, up, GELU, down, added to the residual
            const int rows = std::min(R, N - r0);
            k_layernorm<<<rows, 256>>>(m.d_x + (size_t) r0 * D, L.ln2_w, L.ln2_b, g.eps, D, a16, (int) L.up.kind);
            if (!m.gemm(L.up, a16, rows, m.d_y, VitGpuImpl::dtype(L.up.kind), err)) return false;
            k_bias_gelu_cast<<<rows, 256>>>(m.d_y, L.up_b, F, m.d_f16, (int) L.down.kind);
            if (!m.gemm(L.down, m.d_f16, rows, m.d_y, VitGpuImpl::dtype(L.down.kind), err)) return false;
            k_add_bias_res<<<rows, 256>>>(m.d_x + (size_t) r0 * D, m.d_y, L.down_b, D);
        }
        stage(5);
    }
    if (!sync_err("the layers")) return false;
    // post norm (into the merger's input kind), then the merger on 4-patch rows, in chunks
    const int M = N / 4, RM = std::max(1, R / 4);      // a chunk of RM merged rows is 4 * RM tokens: R rows of the 16-bit input
    for (int r0 = 0; r0 < M; r0 += RM) {
        const int rows = std::min(RM, M - r0);
        k_layernorm<<<rows * 4, 256>>>(m.d_x + (size_t) r0 * 4 * D, m.post_w, m.post_b, g.eps, D, a16, (int) m.mm0.kind);
        if (!m.gemm(m.mm0, a16, rows, m.d_y, VitGpuImpl::dtype(m.mm0.kind), err)) return false;
        k_bias_gelu_cast<<<rows, 256>>>(m.d_y, m.mm0_b, g.merger_hidden, m.d_f16, (int) m.mm2.kind);
        if (!m.gemm(m.mm2, m.d_f16, rows, m.d_out + (size_t) r0 * g.out_dim, VitGpuImpl::dtype(m.mm2.kind), err)) return false;
        k_add_bias<<<rows, 256>>>(m.d_out + (size_t) r0 * g.out_dim, m.mm2_b, g.out_dim);
    }
    if (!sync_err("the projector")) return false;
    stage(6);
    rows.resize((size_t) M * g.out_dim);
    if (cudaMemcpy(rows.data(), m.d_out, rows.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) { err = std::string("GPU copy failed: ") + cudaGetErrorString(cudaGetLastError()); return false; }
    stage(7);
    if (prof)
        std::fprintf(stderr, "vit-prof N=%d: patch-matrix %.2f, patch-embed %.2f, ln+qkv %.2f, attention %.2f (S product %.2f, softmax %.2f), o-proj %.2f, ffn %.2f, merger %.2f, copy-out %.2f ms\n", N, pt[0], pt[1], pt[2],
                     pt[3] + pt[8] + pt[9], pt[8], pt[9], pt[4], pt[5], pt[6], pt[7]);
    return true;
}

}  // namespace strata_vit
