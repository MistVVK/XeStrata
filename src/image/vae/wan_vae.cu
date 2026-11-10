// src/image/vae/wan_vae.cu - Strata's Qwen-Image-2.1 VAE decoder (see include/strata/image/wan_vae.hpp).
//
// Layout: every activation is pixel-major float32, [H][W][C] (a pixel is a row of C channels).  A convolution is
//   gather   -> f16 operand rows for a band of PB output pixels: per pixel the k*k taps x C channels, each value
//               optionally RMS-normed (x * inv_rms(pixel) * gamma) and SiLU'd, or read from the half-size tensor
//               (nearest 2x upsample), then x 2^-7 and rounded to f16 (round to nearest even), zero outside the image;
//   GEMM     -> cuBLAS / hipBLAS GemmEx, f16 x f16, f32 compute, f32 out: [PB][OC] = rows x W^T;
//   epilogue -> x 2^7, + bias, + residual, + the DupUp shortcut, written to the output tensor.
// That is sd.cpp's arithmetic (ggml_ext_conv_2d with scale 1/128: scale, im2col to f16, mul_mat, unscale, bias add)
// except the accumulation, which sd.cpp does in f16 (CUBLAS_COMPUTE_16F) and we do in f32.
// PB is a constant, the last band is padded with zero rows, so every GEMM of a layer has the same shape on every
// picture size: the value of a pixel does not depend on the picture size or the memory of the card.
// The mid-block attention (one head, dim 1152, over all latent pixels) is f32 SGEMM / softmax / SGEMM, like sd.cpp's
// VAE (it does not use flash attention there).
#include "strata/image/wan_vae.hpp"

#include "safetensors_lite.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#if defined(__HIP__) || defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include "strata/hip_compat/cuda_runtime.h"   // the engine's CUDA -> HIP names (cuBLAS -> hipBLAS)
#include "strata/hip_compat/cublas_v2.h"
#else
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace strata::image {
namespace {

constexpr int PB = 8192;                  // pixels per GEMM band (fixed: see the file comment)
constexpr float CONV_SCALE = 0.0078125f;  // 2^-7, wan_vae.hpp "Keep large VAE activations within the FP16 range"
constexpr float CONV_UNSCALE = 128.0f;
constexpr float RMS_EPS = 1e-12f;
constexpr int Z_DIM = 64;

enum GatherMode { G_PLAIN = 0, G_NORM = 1, G_NORM_SILU = 2, G_UP2 = 3 };

#define VAE_CUDA(x)                                                                                        \
    do {                                                                                                   \
        cudaError_t e_ = (x);                                                                              \
        if (e_ != cudaSuccess) return std::string("wan_vae: ") + #x + ": " + cudaGetErrorString(e_);       \
    } while (0)
#define VAE_BLAS(x)                                                                                        \
    do {                                                                                                   \
        auto s_ = (x);                                                                                     \
        if (s_ != CUBLAS_STATUS_SUCCESS) return std::string("wan_vae: ") + #x + ": status " + std::to_string((int) s_); \
    } while (0)
#define VAE_TRY(x)                                                                                         \
    do {                                                                                                   \
        std::string r_ = (x);                                                                              \
        if (!r_.empty()) return r_;                                                                        \
    } while (0)

// ---------------------------------------------------------------------------------------------------------------
// kernels

// inv[p] = 1 / sqrt(mean(x[p][:]^2) + eps); 8 pixels per block of 256 threads, 32 threads per pixel, fixed order.
__global__ void rms_inv_kernel(const float* __restrict__ x, int64_t P, int C, float* __restrict__ inv) {
    __shared__ float part[256];
    const int g = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t p = (int64_t) blockIdx.x * 8 + g;
    float s = 0.f;
    if (p < P) {
        const float* row = x + p * C;
        for (int c = lane; c < C; c += 32) s += row[c] * row[c];
    }
    part[threadIdx.x] = s;
    __syncthreads();
    for (int w = 16; w > 0; w >>= 1) {
        if (lane < w) part[threadIdx.x] += part[threadIdx.x + w];
        __syncthreads();
    }
    if (lane == 0 && p < P) inv[p] = rsqrtf(part[threadIdx.x] / (float) C + RMS_EPS);
}

// One thread = 8 channels of one tap of one band pixel.  out: [PB][taps*C] f16 bits.
template <int MODE>
__global__ void gather_kernel(const float* __restrict__ src, int Hs, int Ws, int C, int H, int W, int k,
                              int64_t p0, int64_t P, const float* __restrict__ inv, const float* __restrict__ gamma,
                              uint16_t* __restrict__ out) {
    const int taps = k * k, c8n = C >> 3;
    const int64_t total = (int64_t) PB * taps * c8n;
    const int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    const int c8 = (int) (t % c8n);
    const int64_t rest = t / c8n;
    const int tap = (int) (rest % taps);
    const int64_t pl = rest / taps;
    const int64_t p = p0 + pl;
    uint16_t h[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (p < P) {
        const int y = (int) (p / W), xq = (int) (p % W);
        const int yy = y + tap / k - k / 2, xx = xq + tap % k - k / 2;
        if (yy >= 0 && yy < H && xx >= 0 && xx < W) {
            const int sy = MODE == G_UP2 ? (yy >> 1) : yy, sx = MODE == G_UP2 ? (xx >> 1) : xx;
            const int64_t sp = (int64_t) sy * Ws + sx;
            const float4* s4 = reinterpret_cast<const float4*>(src + sp * C + c8 * 8);
            const float4 a = s4[0], b = s4[1];
            float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
            const float r = (MODE == G_NORM || MODE == G_NORM_SILU) ? inv[sp] : 1.f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float q = v[i];
                if (MODE == G_NORM || MODE == G_NORM_SILU) q = (q * r) * gamma[c8 * 8 + i];
                if (MODE == G_NORM_SILU) q = q / (1.0f + expf(-q));
                q = q * CONV_SCALE;
                h[i] = strata::kernels::f16_from_f32(q);
            }
        }
        (void) Hs;
    }
    uint4 pk;
    pk.x = (uint32_t) h[0] | ((uint32_t) h[1] << 16);
    pk.y = (uint32_t) h[2] | ((uint32_t) h[3] << 16);
    pk.z = (uint32_t) h[4] | ((uint32_t) h[5] << 16);
    pk.w = (uint32_t) h[6] | ((uint32_t) h[7] << 16);
    *reinterpret_cast<uint4*>(out + pl * ((int64_t) taps * C) + (int64_t) tap * C + c8 * 8) = pk;
}

// dst[p][oc] = ((acc * 2^7 + bias) + res) + dup; dup = the DupUp3D shortcut of x0 [Hs][Ws][C0] for an output of
// [2Hs][2Ws][OC]: channel (oc*ft*4 + (ft-1)*4 + i*2 + j) / repeats of x0's pixel (Y/2, X/2), i = Y&1, j = X&1.
__global__ void epilogue_kernel(const float* __restrict__ acc, int OC, int64_t p0, int64_t nvalid,
                                const float* __restrict__ bias, const float* res, float* dst, int W,
                                const float* __restrict__ x0, int C0, int ft) {
    const int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nvalid * OC) return;
    const int oc = (int) (t % OC);
    const int64_t pl = t / OC, p = p0 + pl;
    float v = acc[pl * OC + oc] * CONV_UNSCALE;
    v = v + bias[oc];
    if (res) v = v + res[p * OC + oc];
    if (x0) {
        const int Y = (int) (p / W), X = (int) (p % W);
        const int repeats = OC * ft * 4 / C0;
        const int ch = (oc * ft * 4 + (ft - 1) * 4 + (Y & 1) * 2 + (X & 1)) / repeats;
        const int64_t sp = (int64_t) (Y >> 1) * (W >> 1) + (X >> 1);
        v = v + x0[sp * C0 + ch];
    }
    dst[p * OC + oc] = v;
}

// Row softmax of s [n][n] in place with the scale applied first (ggml soft_max: max, exp(x - max), sum, x / sum).
__global__ void softmax_rows_kernel(float* s, int n, float scale) {
    __shared__ float red[256];
    float* row = s + (int64_t) blockIdx.x * n;
    float m = -INFINITY;
    for (int j = threadIdx.x; j < n; j += 256) m = fmaxf(m, row[j] * scale);
    red[threadIdx.x] = m;
    __syncthreads();
    for (int w = 128; w > 0; w >>= 1) {
        if (threadIdx.x < w) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + w]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();
    float sum = 0.f;
    for (int j = threadIdx.x; j < n; j += 256) {
        const float e = expf(row[j] * scale - m);
        row[j] = e;
        sum += e;
    }
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int w = 128; w > 0; w >>= 1) {
        if (threadIdx.x < w) red[threadIdx.x] += red[threadIdx.x + w];
        __syncthreads();
    }
    const float inv = 1.0f / red[0];
    for (int j = threadIdx.x; j < n; j += 256) row[j] = row[j] * inv;
}

// ---------------------------------------------------------------------------------------------------------------
// host side

struct Pool {   // device buffers reused by size; peak = what was ever allocated at once
    struct Blk { void* p; size_t n; bool used; };
    std::vector<Blk> blks;
    size_t live = 0, peak = 0;
    void* get(size_t n, std::string& err) {
        Blk* best = nullptr;
        for (auto& b : blks)
            if (!b.used && b.n >= n && (!best || b.n < best->n)) best = &b;
        if (best) { best->used = true; return best->p; }
        void* p = nullptr;
        cudaError_t e = cudaMalloc(&p, n);
        if (e != cudaSuccess) {   // give back the free blocks and try once more
            for (auto it = blks.begin(); it != blks.end();) {
                if (!it->used) { cudaFree(it->p); live -= it->n; it = blks.erase(it); } else ++it;
            }
            e = cudaMalloc(&p, n);
            if (e != cudaSuccess) { err = std::string("wan_vae: out of device memory (") + std::to_string(n >> 20) + " MiB)"; return nullptr; }
        }
        blks.push_back({p, n, true});
        live += n;
        if (live > peak) peak = live;
        return p;
    }
    void put(const void* p) {
        for (auto& b : blks) if (b.p == p) { b.used = false; return; }
    }
    void clear() { for (auto& b : blks) cudaFree(b.p); blks.clear(); live = 0; }
};

struct Tensor { float* d = nullptr; int H = 0, W = 0, C = 0; int64_t P() const { return (int64_t) H * W; } };

struct Conv { int ic = 0, oc = 0, k = 0; uint16_t* w = nullptr; float* b = nullptr; };
struct Norm { int c = 0; float* g = nullptr; };
struct Resnet { Norm n1, n2; Conv c1, c2, sc; bool has_sc = false; };

}  // namespace

struct WanVae {
    int device = 0;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    std::vector<void*> weights;
    size_t weight_bytes = 0;
    Conv post_quant, conv_in, qkv, proj, conv_out;
    Norm attn_norm, norm_out;
    Resnet mid0, mid1;
    Resnet up[5][3];
    Conv resample[5];
    Pool pool;
    uint16_t* gbuf = nullptr;   // [PB][Kmax] f16
    float* abuf = nullptr;      // [PB][OCmax] f32
    std::function<void(const char*, const float*, int, int, int)> dump;
};

namespace {

const int UP_FT[5] = {2, 2, 2, 1, 1};          // DupUp3D temporal factor (temperal_upsample {T, T, T, F})
const bool UP_HAS[5] = {true, true, true, true, false};

std::string upload(WanVae* v, const void* host, size_t n, void** dev) {
    VAE_CUDA(cudaMalloc(dev, n));
    v->weights.push_back(*dev);
    v->weight_bytes += n;
    VAE_CUDA(cudaMemcpy(*dev, host, n, cudaMemcpyHostToDevice));
    return "";
}

struct Loader {
    std::string path;
    StFile file;
    std::string floats(const std::string& name, std::vector<float>& out, std::vector<int64_t>* shape = nullptr) {
        auto it = file.tensors.find(name);
        if (it == file.tensors.end()) return "wan_vae: tensor " + name + " missing in " + path;
        std::vector<unsigned char> raw;
        VAE_TRY(st_read_tensor(path, file, it->second, raw));
        const auto& t = it->second;
        size_t n = 1;
        for (auto s : t.shape) n *= (size_t) s;
        out.resize(n);
        if (t.dtype == "BF16" && raw.size() == n * 2) {
            for (size_t i = 0; i < n; ++i) out[i] = strata::kernels::f32_from_bf16((uint16_t) (raw[2 * i] | (raw[2 * i + 1] << 8)));
        } else if (t.dtype == "F32" && raw.size() == n * 4) {
            std::memcpy(out.data(), raw.data(), n * 4);
        } else {
            return "wan_vae: tensor " + name + " has dtype " + t.dtype + " (bf16 or f32 expected)";
        }
        if (shape) *shape = t.shape;
        return "";
    }
};

std::string load_conv(WanVae* v, Loader& L, const std::string& name, Conv& cv) {
    std::vector<float> w, b;
    std::vector<int64_t> sh;
    VAE_TRY(L.floats(name + ".weight", w, &sh));
    VAE_TRY(L.floats(name + ".bias", b));
    if (sh.size() != 4 || sh[2] != sh[3] || (sh[2] != 1 && sh[2] != 3) || (int64_t) b.size() != sh[0] || sh[1] % 8 != 0)
        return "wan_vae: " + name + ": unexpected shape";
    cv.oc = (int) sh[0];
    cv.ic = (int) sh[1];
    cv.k = (int) sh[2];
    const int taps = cv.k * cv.k, K = taps * cv.ic;
    std::vector<uint16_t> h((size_t) cv.oc * K);   // [oc][(ky*k+kx)*ic + ic], bf16 -> f16 round to nearest even
    for (int o = 0; o < cv.oc; ++o)
        for (int i = 0; i < cv.ic; ++i)
            for (int t = 0; t < taps; ++t)
                h[(size_t) o * K + (size_t) t * cv.ic + i] = strata::kernels::f16_from_f32(w[((size_t) o * cv.ic + i) * taps + t]);
    void* d = nullptr;
    VAE_TRY(upload(v, h.data(), h.size() * 2, &d));
    cv.w = (uint16_t*) d;
    VAE_TRY(upload(v, b.data(), b.size() * 4, &d));
    cv.b = (float*) d;
    return "";
}

std::string load_norm(WanVae* v, Loader& L, const std::string& name, Norm& n) {
    std::vector<float> g;
    VAE_TRY(L.floats(name + ".gamma", g));
    n.c = (int) g.size();
    void* d = nullptr;
    VAE_TRY(upload(v, g.data(), g.size() * 4, &d));
    n.g = (float*) d;
    return "";
}

std::string load_resnet(WanVae* v, Loader& L, const std::string& p, Resnet& r) {
    VAE_TRY(load_norm(v, L, p + ".norm1", r.n1));
    VAE_TRY(load_conv(v, L, p + ".conv1", r.c1));
    VAE_TRY(load_norm(v, L, p + ".norm2", r.n2));
    VAE_TRY(load_conv(v, L, p + ".conv2", r.c2));
    r.has_sc = L.file.tensors.count(p + ".conv_shortcut.weight") > 0;
    if (r.has_sc) VAE_TRY(load_conv(v, L, p + ".conv_shortcut", r.sc));
    return "";
}

std::string alloc_t(WanVae* v, Tensor& t, int H, int W, int C) {
    std::string err;
    t.H = H; t.W = W; t.C = C;
    t.d = (float*) v->pool.get((size_t) H * W * C * sizeof(float), err);
    return t.d ? "" : err;
}
void free_t(WanVae* v, Tensor& t) { if (t.d) v->pool.put(t.d); t.d = nullptr; }

// dst (must be allocated, [H][W][oc]) = conv(src) with the gather mode, + res (may be dst itself), + DupUp(x0).
std::string conv(WanVae* v, const Conv& cv, const Tensor& src, int mode, const Norm* nrm, Tensor& dst,
                 const float* res, const Tensor* x0, int ft) {
    if (src.C != cv.ic) return "wan_vae: conv input channels differ";
    if (nrm && nrm->c != src.C) return "wan_vae: norm channels differ";
    const int H = mode == G_UP2 ? src.H * 2 : src.H, W = mode == G_UP2 ? src.W * 2 : src.W;
    if (dst.H != H || dst.W != W || dst.C != cv.oc) return "wan_vae: conv output shape differs";
    const int64_t P = (int64_t) H * W;
    float* inv = nullptr;
    std::string err;
    if (mode == G_NORM || mode == G_NORM_SILU) {
        inv = (float*) v->pool.get((size_t) src.P() * sizeof(float), err);
        if (!inv) return err;
        rms_inv_kernel<<<(unsigned) ((src.P() + 7) / 8), 256, 0, v->stream>>>(src.d, src.P(), src.C, inv);
        VAE_CUDA(cudaGetLastError());
    }
    const int K = cv.k * cv.k * cv.ic;
    const float one = 1.f, zero = 0.f;
    for (int64_t p0 = 0; p0 < P; p0 += PB) {
        const int64_t total = (int64_t) PB * cv.k * cv.k * (cv.ic / 8);
        const unsigned blocks = (unsigned) ((total + 255) / 256);
        switch (mode) {
            case G_PLAIN: gather_kernel<G_PLAIN><<<blocks, 256, 0, v->stream>>>(src.d, src.H, src.W, src.C, H, W, cv.k, p0, P, inv, nullptr, v->gbuf); break;
            case G_NORM: gather_kernel<G_NORM><<<blocks, 256, 0, v->stream>>>(src.d, src.H, src.W, src.C, H, W, cv.k, p0, P, inv, nrm->g, v->gbuf); break;
            case G_NORM_SILU: gather_kernel<G_NORM_SILU><<<blocks, 256, 0, v->stream>>>(src.d, src.H, src.W, src.C, H, W, cv.k, p0, P, inv, nrm->g, v->gbuf); break;
            default: gather_kernel<G_UP2><<<blocks, 256, 0, v->stream>>>(src.d, src.H, src.W, src.C, H, W, cv.k, p0, P, inv, nullptr, v->gbuf); break;
        }
        VAE_CUDA(cudaGetLastError());
        VAE_BLAS(cublasGemmEx(v->blas, CUBLAS_OP_T, CUBLAS_OP_N, cv.oc, PB, K, &one, cv.w, CUDA_R_16F, K, v->gbuf,
                              CUDA_R_16F, K, &zero, v->abuf, CUDA_R_32F, cv.oc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        const int64_t nvalid = (P - p0) < PB ? (P - p0) : PB;
        const int64_t et = nvalid * cv.oc;
        epilogue_kernel<<<(unsigned) ((et + 255) / 256), 256, 0, v->stream>>>(
            v->abuf, cv.oc, p0, nvalid, cv.b, res, dst.d, W, x0 ? x0->d : nullptr, x0 ? x0->C : 0, ft);
        VAE_CUDA(cudaGetLastError());
    }
    if (inv) v->pool.put(inv);
    return "";
}

// x -> resnet(x); x is consumed (its buffer becomes the output when the channels stay, else it is freed).
std::string resnet(WanVae* v, const Resnet& r, Tensor& x) {
    Tensor h, t;
    if (r.has_sc) {
        VAE_TRY(alloc_t(v, h, x.H, x.W, r.sc.oc));
        VAE_TRY(conv(v, r.sc, x, G_PLAIN, nullptr, h, nullptr, nullptr, 0));
    } else {
        h = x;
    }
    VAE_TRY(alloc_t(v, t, x.H, x.W, r.c1.oc));
    VAE_TRY(conv(v, r.c1, x, G_NORM_SILU, &r.n1, t, nullptr, nullptr, 0));
    VAE_TRY(conv(v, r.c2, t, G_NORM_SILU, &r.n2, h, h.d, nullptr, 0));   // in place on h
    free_t(v, t);
    if (r.has_sc) free_t(v, x);
    x = h;
    return "";
}

std::string attention(WanVae* v, Tensor& x) {
    const int C = x.C;
    const int64_t n = x.P();
    Tensor qkv, o;
    VAE_TRY(alloc_t(v, qkv, x.H, x.W, 3 * C));
    VAE_TRY(conv(v, v->qkv, x, G_NORM, &v->attn_norm, qkv, nullptr, nullptr, 0));
    std::string err;
    float* s = (float*) v->pool.get((size_t) n * n * sizeof(float), err);
    if (!s) return err;
    const float one = 1.f, zero = 0.f;
    // s (row-major [n][n], row i = query i) = Q K^T: column-major S^T = K * Q^T ... with ld 3C
    VAE_BLAS(cublasSgemm(v->blas, CUBLAS_OP_T, CUBLAS_OP_N, (int) n, (int) n, C, &one, qkv.d + C, 3 * C, qkv.d, 3 * C,
                         &zero, s, (int) n));
    softmax_rows_kernel<<<(unsigned) n, 256, 0, v->stream>>>(s, (int) n, 1.0f / std::sqrt((float) C));
    VAE_CUDA(cudaGetLastError());
    VAE_TRY(alloc_t(v, o, x.H, x.W, C));
    // o (row-major [n][C]) = A V: column-major O (C x n) = V (C x n, ld 3C) * A^T-view (n x n, ld n)
    VAE_BLAS(cublasSgemm(v->blas, CUBLAS_OP_N, CUBLAS_OP_N, C, (int) n, (int) n, &one, qkv.d + 2 * C, 3 * C, s, (int) n,
                         &zero, o.d, C));
    v->pool.put(s);
    free_t(v, qkv);
    VAE_TRY(conv(v, v->proj, o, G_PLAIN, nullptr, x, x.d, nullptr, 0));       // proj + identity, in place on x
    free_t(v, o);
    return "";
}

std::string stage(WanVae* v, const char* name, const Tensor& t) {
    if (!v->dump) return "";
    std::vector<float> h((size_t) t.P() * t.C);
    VAE_CUDA(cudaMemcpyAsync(h.data(), t.d, h.size() * 4, cudaMemcpyDeviceToHost, v->stream));
    VAE_CUDA(cudaStreamSynchronize(v->stream));
    v->dump(name, h.data(), t.H, t.W, t.C);
    return "";
}

std::string decode_impl(WanVae* v, const float* latent, int h, int w, float* out, WanVaeStats* st) {
    VAE_CUDA(cudaSetDevice(v->device));
    cudaEvent_t e0, e1;
    VAE_CUDA(cudaEventCreate(&e0));
    VAE_CUDA(cudaEventCreate(&e1));
    VAE_CUDA(cudaEventRecord(e0, v->stream));
    Tensor z, x;
    {
        std::vector<float> hwc((size_t) h * w * Z_DIM);
        for (int c = 0; c < Z_DIM; ++c)
            for (int i = 0; i < h * w; ++i) hwc[(size_t) i * Z_DIM + c] = latent[(size_t) c * h * w + i];
        VAE_TRY(alloc_t(v, z, h, w, Z_DIM));
        VAE_CUDA(cudaMemcpyAsync(z.d, hwc.data(), hwc.size() * 4, cudaMemcpyHostToDevice, v->stream));
        VAE_CUDA(cudaStreamSynchronize(v->stream));
    }
    Tensor y;
    VAE_TRY(alloc_t(v, y, h, w, v->post_quant.oc));
    VAE_TRY(conv(v, v->post_quant, z, G_PLAIN, nullptr, y, nullptr, nullptr, 0));
    free_t(v, z);
    VAE_TRY(alloc_t(v, x, h, w, v->conv_in.oc));
    VAE_TRY(conv(v, v->conv_in, y, G_PLAIN, nullptr, x, nullptr, nullptr, 0));
    free_t(v, y);
    VAE_TRY(stage(v, "conv_in", x));
    VAE_TRY(resnet(v, v->mid0, x));
    VAE_TRY(attention(v, x));
    VAE_TRY(stage(v, "mid_attn", x));
    VAE_TRY(resnet(v, v->mid1, x));
    VAE_TRY(stage(v, "mid", x));
    for (int i = 0; i < 5; ++i) {
        Tensor x0;
        if (UP_HAS[i]) {   // keep the block's input for the DupUp shortcut
            VAE_TRY(alloc_t(v, x0, x.H, x.W, x.C));
            VAE_CUDA(cudaMemcpyAsync(x0.d, x.d, (size_t) x.P() * x.C * 4, cudaMemcpyDeviceToDevice, v->stream));
        }
        for (int j = 0; j < 3; ++j) VAE_TRY(resnet(v, v->up[i][j], x));
        if (UP_HAS[i]) {
            Tensor u;
            VAE_TRY(alloc_t(v, u, x.H * 2, x.W * 2, v->resample[i].oc));
            VAE_TRY(conv(v, v->resample[i], x, G_UP2, nullptr, u, nullptr, &x0, UP_FT[i]));
            free_t(v, x);
            free_t(v, x0);
            x = u;
        }
        const char* names[5] = {"up0", "up1", "up2", "up3", "up4"};
        VAE_TRY(stage(v, names[i], x));
    }
    Tensor o;
    VAE_TRY(alloc_t(v, o, x.H, x.W, v->conv_out.oc));
    VAE_TRY(conv(v, v->conv_out, x, G_NORM_SILU, &v->norm_out, o, nullptr, nullptr, 0));
    free_t(v, x);
    VAE_TRY(stage(v, "out", o));
    std::vector<float> hwc((size_t) o.P() * o.C);
    VAE_CUDA(cudaMemcpyAsync(hwc.data(), o.d, hwc.size() * 4, cudaMemcpyDeviceToHost, v->stream));
    VAE_CUDA(cudaEventRecord(e1, v->stream));
    VAE_CUDA(cudaEventSynchronize(e1));
    free_t(v, o);
    const int64_t P = o.P();
    for (int c = 0; c < o.C; ++c)
        for (int64_t i = 0; i < P; ++i) out[(size_t) c * P + i] = hwc[(size_t) i * o.C + c];
    float ms = 0;
    VAE_CUDA(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    if (st) {
        st->decode_ms = ms;
        st->weight_bytes = v->weight_bytes;
        st->peak_bytes = v->weight_bytes + v->pool.peak;
    }
    return "";
}

std::string open_impl(WanVae* v, const std::string& path) {
    Loader L;
    L.path = path;
    VAE_TRY(st_read_header(path, L.file));
    VAE_CUDA(cudaSetDevice(v->device));
    VAE_CUDA(cudaStreamCreateWithFlags(&v->stream, cudaStreamNonBlocking));
    VAE_BLAS(cublasCreate(&v->blas));
    VAE_BLAS(cublasSetStream(v->blas, v->stream));
    VAE_TRY(load_conv(v, L, "post_quant_conv", v->post_quant));
    VAE_TRY(load_conv(v, L, "decoder.conv_in", v->conv_in));
    VAE_TRY(load_resnet(v, L, "decoder.mid_block.resnets.0", v->mid0));
    VAE_TRY(load_norm(v, L, "decoder.mid_block.attentions.0.norm", v->attn_norm));
    VAE_TRY(load_conv(v, L, "decoder.mid_block.attentions.0.to_qkv", v->qkv));
    VAE_TRY(load_conv(v, L, "decoder.mid_block.attentions.0.proj", v->proj));
    VAE_TRY(load_resnet(v, L, "decoder.mid_block.resnets.1", v->mid1));
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 3; ++j)
            VAE_TRY(load_resnet(v, L, "decoder.up_blocks." + std::to_string(i) + ".resnets." + std::to_string(j), v->up[i][j]));
        if (UP_HAS[i]) VAE_TRY(load_conv(v, L, "decoder.up_blocks." + std::to_string(i) + ".upsampler.resample.1", v->resample[i]));
    }
    VAE_TRY(load_norm(v, L, "decoder.norm_out", v->norm_out));
    VAE_TRY(load_conv(v, L, "decoder.conv_out", v->conv_out));
    if (v->post_quant.ic != Z_DIM || v->conv_in.ic != Z_DIM || v->conv_out.oc != 4)
        return "wan_vae: " + path + " is not a Qwen-Image-2.1 VAE (latent channels / output channels differ)";
    // the work buffers: the largest K and OC of any layer
    int kmax = 0, ocmax = 0;
    auto see = [&](const Conv& c) { kmax = std::max(kmax, c.k * c.k * c.ic); ocmax = std::max(ocmax, c.oc); };
    see(v->post_quant); see(v->conv_in); see(v->qkv); see(v->proj); see(v->conv_out);
    for (const Resnet* r : {&v->mid0, &v->mid1}) { see(r->c1); see(r->c2); if (r->has_sc) see(r->sc); }
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 3; ++j) { see(v->up[i][j].c1); see(v->up[i][j].c2); if (v->up[i][j].has_sc) see(v->up[i][j].sc); }
        if (UP_HAS[i]) see(v->resample[i]);
    }
    void* d = nullptr;
    VAE_CUDA(cudaMalloc(&d, (size_t) PB * kmax * 2));
    v->weights.push_back(d);
    v->gbuf = (uint16_t*) d;
    VAE_CUDA(cudaMalloc(&d, (size_t) PB * ocmax * 4));
    v->weights.push_back(d);
    v->abuf = (float*) d;
    v->weight_bytes += (size_t) PB * kmax * 2 + (size_t) PB * ocmax * 4;
    return "";
}

}  // namespace

WanVae* wan_vae_open(const std::string& path, int device, std::string* err) {
    auto* v = new WanVae();
    if (device < 0) {
        if (cudaGetDevice(&device) != cudaSuccess) device = 0;
    }
    v->device = device;
    std::string r = open_impl(v, path);
    if (!r.empty()) {
        if (err) *err = r;
        wan_vae_close(v);
        return nullptr;
    }
    return v;
}

std::string wan_vae_decode(WanVae* v, const float* latent, int h, int w, float* out, WanVaeStats* stats) {
    if (!v || !latent || !out || h <= 0 || w <= 0) return "wan_vae: bad arguments";
    std::string r = decode_impl(v, latent, h, w, out, stats);
    cudaStreamSynchronize(v->stream);
    for (auto& b : v->pool.blks) b.used = false;   // after an error part-way, nothing stays held
    return r;
}

void wan_vae_denormalize(float* latent, int h, int w) {
    // wan_vae.hpp get_latents_mean_std, VERSION_QWEN_IMAGE_2_1 (64 channels); diffusion_to_vae_latents computes
    // (latents * std) / scale_factor + mean, three float operations, scale_factor 1
    static const float mean[64] = {
        0.5126f, 0.7721f, -0.0631f, 1.3506f, -0.7855f, -2.1025f, -0.3458f, 1.3722f, 1.8873f, -1.7177f, -0.6510f, 0.2732f,
        0.7562f, -0.6163f, -1.0277f, 3.8363f, 2.0210f, 0.0472f, 0.9320f, 2.0087f, 2.4954f, -0.1391f, -1.4249f, 1.8464f,
        -0.5236f, 1.2826f, 3.7046f, -1.3035f, 2.7286f, -1.4518f, -1.9036f, -1.9955f, -0.0342f, -1.0265f, -0.7636f, 3.0555f,
        0.0746f, -3.0751f, -0.1076f, 1.7376f, -1.0914f, -1.9435f, -0.2784f, -1.3680f, 0.4809f, -0.4433f, 0.3764f, 0.5729f,
        -2.0595f, 1.0960f, -1.3260f, -2.0211f, -5.0179f, 0.5275f, 4.0162f, 1.8505f, 0.3026f, 1.9373f, 1.4937f, 0.2632f,
        0.5547f, -1.7121f, -0.1562f, 0.0304f};
    static const float stdv[64] = {
        3.2001f, 3.2936f, 3.4321f, 3.0091f, 3.1061f, 4.0379f, 4.0705f, 3.7910f, 3.0785f, 3.6500f, 3.9308f, 3.0904f,
        2.8778f, 3.7675f, 3.7320f, 5.0756f, 3.2864f, 4.0397f, 3.1317f, 4.0443f, 2.9249f, 3.9454f, 3.0988f, 4.2489f,
        3.4896f, 3.8513f, 3.9323f, 3.4719f, 3.7498f, 4.2830f, 3.5694f, 4.2467f, 3.9037f, 3.2947f, 5.0770f, 3.5075f,
        3.2700f, 3.4767f, 2.8063f, 5.1125f, 3.5327f, 4.7833f, 3.1286f, 4.1819f, 3.8527f, 3.8312f, 3.5605f, 4.3875f,
        3.9624f, 4.0168f, 3.5643f, 4.0550f, 5.5614f, 4.2963f, 4.4080f, 3.4959f, 3.8747f, 3.7608f, 3.5735f, 3.1490f,
        3.7662f, 3.6746f, 3.4563f, 3.8161f};
    volatile float one = 1.0f;   // keep the three operations separate, as sd.cpp's Tensor operators are
    for (int c = 0; c < 64; ++c)
        for (int i = 0; i < h * w; ++i) {
            float& x = latent[(size_t) c * h * w + i];
            float a = x * stdv[c];
            a = a / one;
            x = a + mean[c];
        }
}

void wan_vae_set_dump(WanVae* v,std::function<void(const char*, const float*, int, int, int)> dump) {
    if (v) v->dump = std::move(dump);
}

void wan_vae_close(WanVae* v) {
    if (!v) return;
    if (v->stream) cudaStreamSynchronize(v->stream);
    v->pool.clear();
    for (void* p : v->weights) cudaFree(p);
    if (v->blas) cublasDestroy(v->blas);
    if (v->stream) cudaStreamDestroy(v->stream);
    delete v;
}

}  // namespace strata::image
