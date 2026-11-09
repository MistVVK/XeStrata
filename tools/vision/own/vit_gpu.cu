// tools/vision/own/vit_gpu.cu - see vit_gpu.h.  One source for CUDA and HIP (STRATA_VIT_HIP maps the few runtime / BLAS names).
#include "vit_gpu.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(STRATA_VIT_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
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

enum Kind { K_F32 = 0, K_F16 = 1, K_BF16 = 2 };   // the 16-bit kind a GEMM's input takes (its weight's)

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

// A[t][e]: the patch of token t (2x2-merge order), channel-major, e = c * P * P + ky * P + kx, from the normalised CHW picture
__global__ void k_im2col(const float* __restrict__ chw, int W, int H, int P, int nbx, float* __restrict__ out) {
    const int t = blockIdx.x, K = 3 * P * P;
    const int b = t / 4, r = t % 4;
    const int py = 2 * (b / nbx) + r / 2, px = 2 * (b % nbx) + r % 2;
    for (int e = threadIdx.x; e < K; e += blockDim.x) {
        const int c = e / (P * P), q = e % (P * P), ky = q / P, kx = q % P;
        out[(size_t) t * K + e] = chw[(size_t) c * W * H + (size_t) (py * P + ky) * W + (size_t) px * P + kx];
    }
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

// y [N][3D] (the fused qkv product) + bias -> q, k (rotary applied), v, each [N][D].  Pair j of a head is (j, j + hd/2); j < hd/4 turns with the
// token's row, the others with its column.
__global__ void k_qkv_rope(const float* __restrict__ y, const float* __restrict__ bias, int pw, int nbx, int heads, int hd, float theta_scale,
                           float* __restrict__ q, float* __restrict__ k, float* __restrict__ v) {
    const int t = blockIdx.x, D = heads * hd, half = hd / 2, quarter = hd / 4;
    const int b = t / 4, r = t % 4;
    const int ty = 2 * (b / nbx) + r / 2, tx = 2 * (b % nbx) + r % 2;
    const float* yr = y + (size_t) t * 3 * D;
    for (int i = threadIdx.x; i < 2 * heads * half; i += blockDim.x) {
        const int part = i / (heads * half), rem = i % (heads * half), h = rem / half, j = rem % half;
        const int base = part * D + h * hd;
        const float a = yr[base + j] + bias[base + j], c = yr[base + j + half] + bias[base + j + half];
        const float freq = powf(theta_scale, (float) (j % quarter));
        float sn, cs;
        sincosf((float) (j < quarter ? ty : tx) * freq, &sn, &cs);
        float* dst = (part == 0 ? q : k) + (size_t) t * D + h * hd;
        dst[j] = a * cs - c * sn;
        dst[j + half] = a * sn + c * cs;
    }
    for (int e = threadIdx.x; e < D; e += blockDim.x) v[(size_t) t * D + e] = yr[2 * D + e] + bias[2 * D + e];
}

// global attention, online softmax, float32.  A block: 32 queries of one head; 4 threads per query (8 keys of each 32-key tile, hd/4 output dims).
constexpr int BQ = 32, BK = 32;
__global__ void __launch_bounds__(128) k_attention(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, int N, int heads, int hd,
                                                   float scale, float* __restrict__ ctx) {
    extern __shared__ float sm[];
    float* Qs = sm;                       // [BQ][hd]
    float* Ks = Qs + BQ * hd;             // [BK][hd]
    float* Vs = Ks + BK * hd;             // [BK][hd]
    float* Ps = Vs + BK * hd;             // [BQ][BK]
    const int D = heads * hd, h = blockIdx.y, q0 = blockIdx.x * BQ;
    const int qi = threadIdx.x >> 2, sub = threadIdx.x & 3, dpt = hd / 4;
    for (int i = threadIdx.x; i < BQ * hd; i += blockDim.x) {
        const int r = i / hd, d = i % hd;
        Qs[i] = q0 + r < N ? q[(size_t) (q0 + r) * D + h * hd + d] : 0.0f;
    }
    float acc[32];
    for (int d = 0; d < dpt; ++d) acc[d] = 0.0f;
    float m = -FLT_MAX, l = 0.0f;
    for (int j0 = 0; j0 < N; j0 += BK) {
        __syncthreads();
        for (int i = threadIdx.x; i < BK * hd; i += blockDim.x) {
            const int r = i / hd, d = i % hd;
            const bool ok = j0 + r < N;
            Ks[i] = ok ? k[(size_t) (j0 + r) * D + h * hd + d] : 0.0f;
            Vs[i] = ok ? v[(size_t) (j0 + r) * D + h * hd + d] : 0.0f;
        }
        __syncthreads();
        float s[8];
        float tmax = -FLT_MAX;
        const float* qr = Qs + qi * hd;
        for (int kk = 0; kk < 8; ++kk) {
            const int key = sub * 8 + kk;
            const float* kr = Ks + key * hd;
            float a = 0.0f;
            for (int d = 0; d < hd; ++d) a += qr[d] * kr[d];
            s[kk] = j0 + key < N ? a * scale : -FLT_MAX;
            tmax = fmaxf(tmax, s[kk]);
        }
        tmax = fmaxf(tmax, VIT_SHFL_XOR(tmax, 1));
        tmax = fmaxf(tmax, VIT_SHFL_XOR(tmax, 2));
        const float m_new = fmaxf(m, tmax), corr = expf(m - m_new);
        float psum = 0.0f;
        for (int kk = 0; kk < 8; ++kk) {
            const float p = s[kk] == -FLT_MAX ? 0.0f : expf(s[kk] - m_new);
            Ps[qi * BK + sub * 8 + kk] = p;
            psum += p;
        }
        l = l * corr + psum;
        m = m_new;
        __syncthreads();   // the four threads of a query read each other's probabilities next (every thread of the block gets here)
        for (int d = 0; d < dpt; ++d) acc[d] *= corr;
        for (int key = 0; key < BK; ++key) {
            const float p = Ps[qi * BK + key];
            const float* vr = Vs + key * hd + sub * dpt;
            for (int d = 0; d < dpt; ++d) acc[d] += p * vr[d];
        }
    }
    l += VIT_SHFL_XOR(l, 1);
    l += VIT_SHFL_XOR(l, 2);
    if (q0 + qi < N) {
        const float inv = 1.0f / l;
        float* o = ctx + (size_t) (q0 + qi) * D + h * hd + sub * dpt;
        for (int d = 0; d < dpt; ++d) o[d] = acc[d] * inv;
    }
}

// P[r][j] = softmax over j of S[r][j] * scale, one block per row of n scores; P in F16
__global__ void k_softmax_rows(const float* __restrict__ S, int n, float scale, uint16_t* __restrict__ P) {
    const float* s = S + (size_t) blockIdx.x * n;
    uint16_t* p = P + (size_t) blockIdx.x * n;
    __shared__ float red[32];
    const int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize, nw = (blockDim.x + warpSize - 1) / warpSize;
    float m = -FLT_MAX;
    for (int j = threadIdx.x; j < n; j += blockDim.x) m = fmaxf(m, s[j] * scale);
    for (int o = warpSize / 2; o > 0; o >>= 1) m = fmaxf(m, VIT_SHFL_XOR(m, o));
    if (lane == 0) red[wid] = m;
    __syncthreads();
    m = lane < nw ? red[lane] : -FLT_MAX;
    for (int o = warpSize / 2; o > 0; o >>= 1) m = fmaxf(m, VIT_SHFL_XOR(m, o));
    __syncthreads();
    float sum = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) sum += expf(s[j] * scale - m);
    for (int o = warpSize / 2; o > 0; o >>= 1) sum += VIT_SHFL_XOR(sum, o);
    if (lane == 0) red[wid] = sum;
    __syncthreads();
    sum = lane < nw ? red[lane] : 0.0f;
    for (int o = warpSize / 2; o > 0; o >>= 1) sum += VIT_SHFL_XOR(sum, o);
    const float inv = 1.0f / sum;
    for (int j = threadIdx.x; j < n; j += blockDim.x) p[j] = f_to_f16bits(expf(s[j] * scale - m) * inv);
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
    float *d_chw = nullptr, *d_im = nullptr, *d_x = nullptr, *d_y = nullptr, *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_ctx = nullptr, *d_out = nullptr;
    void* d_a = nullptr;            // the GEMM input in its 16-bit kind
    size_t chw_cap = 0;
    uint16_t *d_q16 = nullptr, *d_k16 = nullptr, *d_v16 = nullptr, *d_P = nullptr;   // q, k, v and the probabilities, F16 (as ggml's flash attention holds them)
    float* d_S = nullptr;
    size_t sp_elems = 0;
    static constexpr size_t kScoreElems = (size_t) 64 << 20;   // 64M scores per chunk: 256 MB of float32 + 128 MB of F16
    bool tiled_attention = false;   // STRATA_VIT_ATTN=tiled: the one-kernel float32 attention (slower; kept as a cross-check)

    // C = alpha * op(A) * op(B), batched, column-major as the BLAS wants it
    bool bgemm(bool ta, bool tb, int m, int n, int k, const void* A, cudaDataType_t at, int lda, long long sa, const void* B, cudaDataType_t bt, int ldb,
               long long sb, void* C, cudaDataType_t ct, int ldc, long long sc, int batch, std::string& err) {
        const float alpha = 1.0f, beta = 0.0f;
        const cublasStatus_t s = cublasGemmStridedBatchedEx(blas, ta ? CUBLAS_OP_T : CUBLAS_OP_N, tb ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha, A, at, lda, sa,
                                                            B, bt, ldb, sb, &beta, C, ct, ldc, sc, batch, VIT_COMPUTE_32F, VIT_GEMM_DEFAULT);
        if (s != CUBLAS_STATUS_SUCCESS) { err = "the batched BLAS product failed (status " + std::to_string((int) s) + ")"; return false; }
        return true;
    }

    ~VitGpuImpl() {
        (void) cudaSetDevice(device);
        for (void* p : allocs) (void) cudaFree(p);
        if (d_chw) (void) cudaFree(d_chw);
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
        if (t->type == GT_F16 || t->type == GT_BF16) {          // as stored
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
    static cudaDataType_t dtype(Kind k) { return k == K_F32 ? CUDA_R_32F : k == K_F16 ? CUDA_R_16F : CUDA_R_16BF; }

    // y[M][N] (float32) = A[M][K] (kind of w) * w^T
    bool gemm(const DevW& w, const void* A, int M, void* y, cudaDataType_t kind_a, std::string& err) {
        const float alpha = 1.0f, beta = 0.0f;
        const cublasStatus_t s = cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, w.rows, M, w.cols, &alpha, w.p, kind_a, w.cols, A, kind_a, w.cols, &beta, y,
                                              CUDA_R_32F, w.rows, VIT_COMPUTE_32F, VIT_GEMM_DEFAULT);
        if (s != CUBLAS_STATUS_SUCCESS) { err = "the BLAS matrix product failed (status " + std::to_string((int) s) + ")"; return false; }
        return true;
    }
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
        d_im = d_x = d_y = d_q = d_k = d_v = d_ctx = d_out = nullptr;
        d_q16 = d_k16 = d_v16 = d_P = nullptr;
        d_S = nullptr;
        d_a = nullptr;
        const size_t a_elems = std::max(N * D, N * F);   // the 16-bit GEMM input (the float32 patch matrix is d_im)
        uint16_t* a16 = nullptr;
        // attention by batched GEMMs: scores S (float32) and probabilities P (F16) of a chunk of queries for every head, at most kScoreElems each
        sp_elems = std::min<size_t>((size_t) g.heads * N * N, kScoreElems);
        bool ok = alloc(d_im, N * K0, err, false) && alloc(d_x, N * D, err, false) && alloc(d_y, N * std::max<size_t>(3 * D, F), err, false) &&
                  alloc(d_q, N * D, err, false) && alloc(d_k, N * D, err, false) && alloc(d_v, N * D, err, false) && alloc(d_ctx, N * D, err, false) &&
                  alloc(a16, a_elems, err, false) && alloc(d_out, (N / 4) * (size_t) g.out_dim, err, false) &&
                  alloc(d_q16, N * D, err, false) && alloc(d_k16, N * D, err, false) && alloc(d_v16, N * D, err, false) &&
                  alloc(d_S, sp_elems, err, false) && alloc(d_P, sp_elems, err, false);
        d_a = a16;
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

bool VitGpu::load(const Gguf& gg, const VitGeometry& g, int device, int max_patches, std::string& err) {
    VitGpuImpl& m = *impl_;
    std::string why;
    if (!vit_geometry_supported(g, why)) { err = "this tower is not supported: " + why; return false; }
    m.g = g;
    m.device = device;
    if (cudaSetDevice(device) != cudaSuccess) { err = std::string("cannot use GPU ") + std::to_string(device) + ": " + cudaGetErrorString(cudaGetLastError()); return false; }
    if (cublasCreate(&m.blas) != CUBLAS_STATUS_SUCCESS) { err = "cannot create the BLAS handle"; return false; }
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
    if (const char* e = std::getenv("STRATA_VIT_ATTN")) m.tiled_attention = std::strcmp(e, "tiled") == 0;
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
    const size_t chw_n = img.chw.size();
    if (chw_n > m.chw_cap) {   // the picture's own buffer is small next to the work buffers; kept apart so the warm-up's size is not needed for it
        if (m.d_chw) { (void) cudaFree(m.d_chw); }
        m.d_chw = nullptr;
        void* q = nullptr;
        if (cudaMalloc(&q, chw_n * 4) != cudaSuccess) { (void) cudaGetLastError(); err = "GPU memory: cannot hold the picture"; return false; }
        m.d_chw = (float*) q;
        m.chw_cap = chw_n;
    }
    auto sync_err = [&](const char* what) {
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { err = std::string(what) + ": " + cudaGetErrorString(e); return false; }
        return true;
    };
    if (cudaMemcpy(m.d_chw, img.chw.data(), chw_n * 4, cudaMemcpyHostToDevice) != cudaSuccess) { err = "GPU copy failed"; return false; }
    const int K0 = 3 * P * P;
    // patch embedding (float32 GEMM), then bias + position
    k_im2col<<<N, 256>>>(m.d_chw, img.w, img.h, P, nbx, m.d_im);
    DevW pw_w;
    pw_w.p = m.patch_w; pw_w.kind = K_F32; pw_w.rows = D; pw_w.cols = K0;
    if (!m.gemm(pw_w, m.d_im, N, m.d_x, CUDA_R_32F, err)) return false;
    k_patch_post<<<N, 256>>>(m.d_x, m.patch_b, m.pos, g.pos_grid, pw, ph, nbx, D);
    if (!sync_err("patch embedding")) return false;

    const float theta_scale = powf(g.rope_theta, -2.0f / (float) (HD / 2));
    const float kq_scale = 1.0f / sqrtf((float) HD);
    const size_t attn_smem = (size_t) (BQ * HD + 2 * BK * HD + BQ * BK) * sizeof(float);
    for (const DevLayer& L : m.layers) {
        k_layernorm<<<N, 256>>>(m.d_x, L.ln1_w, L.ln1_b, g.eps, D, m.d_a, (int) L.qkv.kind);
        if (!m.gemm(L.qkv, m.d_a, N, m.d_y, VitGpuImpl::dtype(L.qkv.kind), err)) return false;
        k_qkv_rope<<<N, 256>>>(m.d_y, L.qkv_b, pw, nbx, H, HD, theta_scale, m.d_q, m.d_k, m.d_v);
        if (m.tiled_attention) {
            k_attention<<<dim3((N + BQ - 1) / BQ, H), 128, attn_smem>>>(m.d_q, m.d_k, m.d_v, N, H, HD, kq_scale, m.d_ctx);
        } else {
            // q, k, v as F16, then per chunk of queries: S = K Q^T (one batched product over the heads), softmax, O = V P (a head's matrix is a column
            // slice of the [N][D] arrays: leading dimension D, batch stride HD)
            const size_t qkv_n = (size_t) N * D;
            k_cast<<<VitGpuImpl::blocks_for(qkv_n), 256>>>(m.d_q, qkv_n, m.d_q16, K_F16);
            k_cast<<<VitGpuImpl::blocks_for(qkv_n), 256>>>(m.d_k, qkv_n, m.d_k16, K_F16);
            k_cast<<<VitGpuImpl::blocks_for(qkv_n), 256>>>(m.d_v, qkv_n, m.d_v16, K_F16);
            const int QC = (int) std::max<size_t>(1, std::min<size_t>((size_t) N, m.sp_elems / ((size_t) H * N)));
            for (int q0 = 0; q0 < N; q0 += QC) {
                const int qc = std::min(QC, N - q0);
                if (!m.bgemm(true, false, N, qc, HD, m.d_k16, CUDA_R_16F, D, HD, m.d_q16 + (size_t) q0 * D, CUDA_R_16F, D, HD, m.d_S, CUDA_R_32F, N,
                             (long long) qc * N, H, err))
                    return false;
                k_softmax_rows<<<H * qc, 256>>>(m.d_S, N, kq_scale, m.d_P);
                if (!m.bgemm(false, false, HD, qc, N, m.d_v16, CUDA_R_16F, D, HD, m.d_P, CUDA_R_16F, N, (long long) qc * N, m.d_ctx + (size_t) q0 * D,
                             CUDA_R_32F, D, HD, H, err))
                    return false;
            }
        }
        k_cast<<<VitGpuImpl::blocks_for((size_t) N * D), 256>>>(m.d_ctx, (size_t) N * D, m.d_a, (int) L.o.kind);
        if (!m.gemm(L.o, m.d_a, N, m.d_y, VitGpuImpl::dtype(L.o.kind), err)) return false;
        k_add_bias_res<<<N, 256>>>(m.d_x, m.d_y, L.o_b, D);
        k_layernorm<<<N, 256>>>(m.d_x, L.ln2_w, L.ln2_b, g.eps, D, m.d_a, (int) L.up.kind);
        if (!m.gemm(L.up, m.d_a, N, m.d_y, VitGpuImpl::dtype(L.up.kind), err)) return false;
        k_bias_gelu_cast<<<N, 256>>>(m.d_y, L.up_b, F, m.d_a, (int) L.down.kind);
        if (!m.gemm(L.down, m.d_a, N, m.d_y, VitGpuImpl::dtype(L.down.kind), err)) return false;
        k_add_bias_res<<<N, 256>>>(m.d_x, m.d_y, L.down_b, D);
    }
    if (!sync_err("the layers")) return false;
    // post norm (into the merger's input kind), then the merger on 4-patch rows
    const int M = N / 4;
    k_layernorm<<<N, 256>>>(m.d_x, m.post_w, m.post_b, g.eps, D, m.d_a, (int) m.mm0.kind);
    if (!m.gemm(m.mm0, m.d_a, M, m.d_y, VitGpuImpl::dtype(m.mm0.kind), err)) return false;
    k_bias_gelu_cast<<<M, 256>>>(m.d_y, m.mm0_b, g.merger_hidden, m.d_a, (int) m.mm2.kind);
    if (!m.gemm(m.mm2, m.d_a, M, m.d_out, VitGpuImpl::dtype(m.mm2.kind), err)) return false;
    k_add_bias<<<M, 256>>>(m.d_out, m.mm2_b, g.out_dim);
    if (!sync_err("the projector")) return false;
    rows.resize((size_t) M * g.out_dim);
    if (cudaMemcpy(rows.data(), m.d_out, rows.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) { err = std::string("GPU copy failed: ") + cudaGetErrorString(cudaGetLastError()); return false; }
    return true;
}

}  // namespace strata_vit
