// src/image/te/qwen3vl_te.cu - Strata's Qwen-Image-2.1 text encoder (see include/strata/image/qwen3vl_te.hpp).
//
// Layout: activations are token-major, [T][width] (a token is a row).  Per layer:
//   h16  = f16(RMSNorm(x) * attn_norm)                                    rms_f16_kernel
//   qkv  = h16 . [Wq; Wk; Wv]^T          f32 [T][(H + 2 KV) D]           three weights decoded into one f16 buffer, one GemmEx
//   q16, k16 = f16(rope(RMSNorm_head(q|k) * q_norm|k_norm)), v16 = f16(v), head-major [heads][T][D]   qk_rope_kernel
//   per chunk of 256 queries, per query head r of a KV group g (batched over the 8 groups):
//        S = q . k^T (f32), P = f16(softmax(S / sqrt(D), causal)), O = P . v (f32, into att [T][H D])
//   att16 = f16(att);  x += att16 . Wo^T  (GemmEx with beta = 1: straight into the f32 residual)
//   h16  = f16(RMSNorm(x) * ffn_norm);  gu = h16 . [Wgate; Wup]^T  f32 [T][2F]
//   a16  = f16(silu(gate) * up);  x += a16 . Wdown^T
// The rope angles are computed in double on the device (positions are the token indices; Qwen3-VL's IMROPE with
// sections 24/20/20 and equal text positions is the plain NEOX rotation of the two halves), so no host maths library
// enters the result.  Fixed-order reductions everywhere: the same prompt gives the same bytes run to run.
#include "strata/image/qwen3vl_te.hpp"

#include "strata/image/te_dequant.hpp"
#include "strata/kernels/f16_bits.hpp"

#define STRATA_GGUF_MAIN_DISABLED 1
#include "strata/artifact/gguf_reader.hpp"

#if defined(__HIP__) || defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include "strata/hip_compat/cuda_runtime.h"
#include "strata/hip_compat/cublas_v2.h"
#else
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace strata::image {
namespace {

constexpr int QCHUNK = 256;               // queries per attention chunk (fixed: see the file comment)
constexpr int HD = 128;                   // head dim (checked against the file)

#define TE_CUDA(x)                                                                                         \
    do {                                                                                                   \
        cudaError_t e_ = (x);                                                                              \
        if (e_ != cudaSuccess) return std::string("qwen3vl_te: ") + #x + ": " + cudaGetErrorString(e_);    \
    } while (0)
#define TE_BLAS(x)                                                                                         \
    do {                                                                                                   \
        auto s_ = (x);                                                                                     \
        if (s_ != CUBLAS_STATUS_SUCCESS) return std::string("qwen3vl_te: ") + #x + ": status " + std::to_string((int) s_); \
    } while (0)
#define TE_TRY(x)                                                                                          \
    do {                                                                                                   \
        std::string r_ = (x);                                                                              \
        if (!r_.empty()) return r_;                                                                        \
    } while (0)

using strata::kernels::f16_from_f32;

// ---------------------------------------------------------------------------------------------------------------
// kernels (256 threads; fixed-order block reductions)

__device__ __forceinline__ float block_sum256(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float block_max256(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

// out[t][c] = f16(x[t][c] * inv_rms(x[t]) * w[c]); one block per token
__global__ void rms_f16_kernel(const float* __restrict__ x, const float* __restrict__ w, int C, float eps,
                               uint16_t* __restrict__ out) {
    __shared__ float sh[256];
    const float* row = x + (size_t) blockIdx.x * C;
    float s = 0.f;
    for (int c = threadIdx.x; c < C; c += 256) s += row[c] * row[c];
    s = block_sum256(s, sh);
    const float inv = 1.0f / sqrtf(s / (float) C + eps);
    uint16_t* o = out + (size_t) blockIdx.x * C;
    for (int c = threadIdx.x; c < C; c += 256) o[c] = f16_from_f32((row[c] * inv) * w[c]);
}

// rope table: cs[t][i] = (cos, sin) of t * base^(-2i/D), i < D/2, in double
__global__ void rope_table_kernel(int T, double base, float2* __restrict__ cs) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * (HD / 2)) return;
    const int t = idx / (HD / 2), i = idx % (HD / 2);
    const double a = (double) t * pow(base, -2.0 * i / HD);
    cs[idx] = make_float2((float) cos(a), (float) sin(a));
}

// one block of 128 threads per (token, head) of q and k: per-head RMSNorm * w, NEOX rope, f16, head-major out;
// heads H + KV + KV: q heads, k heads, then v heads (only cast)
__global__ void qk_rope_kernel(const float* __restrict__ qkv, int T, int H, int KV, const float* __restrict__ qw,
                               const float* __restrict__ kw, float eps, const float2* __restrict__ cs,
                               uint16_t* __restrict__ q16, uint16_t* __restrict__ k16, uint16_t* __restrict__ v16) {
    __shared__ float sh[HD];
    __shared__ float red[HD];
    const int t = blockIdx.x, hh = blockIdx.y, d = threadIdx.x;
    const int width = (H + 2 * KV) * HD;
    const float v = qkv[(size_t) t * width + (size_t) hh * HD + d];
    if (hh >= H + KV) {                                   // v: cast only
        v16[((size_t) (hh - H - KV) * T + t) * HD + d] = f16_from_f32(v);
        return;
    }
    red[d] = v * v;
    __syncthreads();
    for (int s = HD / 2; s > 0; s >>= 1) {
        if (d < s) red[d] += red[d + s];
        __syncthreads();
    }
    const float inv = 1.0f / sqrtf(red[0] / (float) HD + eps);
    const float* w = hh < H ? qw : kw;
    sh[d] = (v * inv) * w[d];
    __syncthreads();
    const int i = d < HD / 2 ? d : d - HD / 2;
    const float2 c = cs[(size_t) t * (HD / 2) + i];
    const float a = sh[i], b = sh[i + HD / 2];
    const float r = d < HD / 2 ? a * c.x - b * c.y : a * c.y + b * c.x;
    if (hh < H) q16[((size_t) hh * T + t) * HD + d] = f16_from_f32(r);
    else k16[((size_t) (hh - H) * T + t) * HD + d] = f16_from_f32(r);
}

// causal softmax over rows of S (f32 [rows][ld], row j of the chunk is query q0 + (j % qc)), scale first, -> f16 P with
// the same layout; columns past the query are 0.
__global__ void softmax_causal_kernel(const float* __restrict__ S, int ld, int ncols, int qc, int q0, float scale,
                                      uint16_t* __restrict__ P) {
    __shared__ float sh[256];
    const int row = blockIdx.x;
    const int q = q0 + row % qc;
    const int valid = min(q + 1, ncols);
    const float* s = S + (size_t) row * ld;
    float m = -INFINITY;
    for (int c = threadIdx.x; c < valid; c += 256) m = fmaxf(m, s[c] * scale);
    m = block_max256(m, sh);
    float sum = 0.f;
    for (int c = threadIdx.x; c < valid; c += 256) sum += expf(s[c] * scale - m);
    sum = block_sum256(sum, sh);
    const float inv = 1.0f / sum;
    uint16_t* p = P + (size_t) row * ld;
    for (int c = threadIdx.x; c < ncols; c += 256) p[c] = c < valid ? f16_from_f32(expf(s[c] * scale - m) * inv) : (uint16_t) 0;
}

__global__ void swiglu_f16_kernel(const float* __restrict__ gu, int T, int F, uint16_t* __restrict__ out) {
    const size_t n = (size_t) T * F;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const size_t t = i / F, f = i % F;
        const float g = gu[t * 2 * F + f], u = gu[t * 2 * F + F + f];
        out[i] = f16_from_f32((g / (1.0f + expf(-g))) * u);
    }
}

__global__ void cast_f16_kernel(const float* __restrict__ in, size_t n, uint16_t* __restrict__ out) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        out[i] = f16_from_f32(in[i]);
}

unsigned grid_for(size_t n) { return (unsigned) std::min<size_t>((n + 255) / 256, 65535u * 8); }

struct WeightRef { uint32_t type = 0; size_t off = 0; int64_t ne0 = 0, rows = 0; };

struct Layer {
    WeightRef q, k, v, o, gate, up, down;
    size_t attn_norm = 0, ffn_norm = 0, q_norm = 0, k_norm = 0;    // float offsets into the norm buffer
};

}  // namespace

struct Qwen3VlTe {
    std::unique_ptr<strata::GgufFile> file;
    int device = 0;
    int n_layer = 0, C = 0, F = 0, H = 0, KV = 0;
    float eps = 1e-6f;
    double theta = 5e6;
    std::vector<Layer> layers;
    const strata::TensorInfo* embd = nullptr;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    uint8_t* d_w = nullptr;            // every layer weight, raw blocks
    float* d_norm = nullptr;           // every 1-D norm weight, f32
    uint16_t* d_stage = nullptr;       // one decoded f16 weight group ([Wq; Wk; Wv] or [Wgate; Wup] or Wo / Wdown)
    size_t stage_elems = 0;
    size_t weight_bytes = 0, norm_bytes = 0;
    double read_ms = 0, upload_ms = 0;
};

namespace {

std::string find_weight(Qwen3VlTe* te, const std::string& name, int64_t ne0, int64_t rows, WeightRef& w, size_t& off,
                        std::vector<std::pair<const strata::TensorInfo*, size_t>>& plan) {
    const strata::TensorInfo* t = te->file->find(name);
    if (!t) return "qwen3vl_te: tensor " + name + " missing";
    if (t->shape.size() != 2 || (int64_t) t->shape[0] != ne0 || (int64_t) t->shape[1] != rows)
        return "qwen3vl_te: tensor " + name + " has an unexpected shape";
    const size_t rb = te_row_bytes(t->type, ne0);
    if (rb == 0) return "qwen3vl_te: tensor " + name + " has type " + t->type_name() + ", which the encoder does not read";
    w = {t->type, off, ne0, rows};
    plan.emplace_back(t, off);
    off += (rb * rows + 255) / 256 * 256;
    return std::string();
}

std::string find_norm(Qwen3VlTe* te, const std::string& name, int64_t n, size_t& at, std::vector<float>& host) {
    const strata::TensorInfo* t = te->file->find(name);
    if (!t) return "qwen3vl_te: tensor " + name + " missing";
    if ((int64_t) t->elements() != n) return "qwen3vl_te: tensor " + name + " has an unexpected size";
    at = host.size();
    host.resize(at + (size_t) n);
    return te_dequant_f32_host(t->type, te->file->tensor_data(*t), n, 1, host.data() + at);
}

std::string meta_int(const strata::GgufFile& f, const std::string& key, int& v) {
    const strata::MetaValue* m = f.get(key);
    if (!m || !m->is_num()) return "qwen3vl_te: metadata " + key + " missing";
    v = (int) m->num();
    return std::string();
}

std::string open_impl(Qwen3VlTe* te, const std::string& path, int device) {
    const auto t0 = std::chrono::steady_clock::now();
    te->file = std::make_unique<strata::GgufFile>(path);
    const strata::GgufFile& f = *te->file;
    const strata::MetaValue* arch = f.get("general.architecture");
    if (!arch || arch->s != "qwen3vl") return "qwen3vl_te: " + path + " is not a qwen3vl GGUF";
    int hd = 0;
    TE_TRY(meta_int(f, "qwen3vl.block_count", te->n_layer));
    TE_TRY(meta_int(f, "qwen3vl.embedding_length", te->C));
    TE_TRY(meta_int(f, "qwen3vl.feed_forward_length", te->F));
    TE_TRY(meta_int(f, "qwen3vl.attention.head_count", te->H));
    TE_TRY(meta_int(f, "qwen3vl.attention.head_count_kv", te->KV));
    TE_TRY(meta_int(f, "qwen3vl.attention.key_length", hd));
    if (const strata::MetaValue* m = f.get("qwen3vl.rope.freq_base")) te->theta = m->num();
    if (hd != HD || te->H * HD != te->C || te->H % te->KV != 0) return "qwen3vl_te: unexpected attention geometry";
    te->eps = 1e-6f;   // sd.cpp llm.hpp: config.rms_norm_eps for QWEN3_VL (the file says 9.99999997e-07, the same f32)
    te->embd = f.find("token_embd.weight");
    if (!te->embd || te_row_bytes(te->embd->type, te->C) == 0) return "qwen3vl_te: token_embd.weight missing or unreadable";

    const int C = te->C, F = te->F, Q = te->H * HD, K = te->KV * HD;
    std::vector<std::pair<const strata::TensorInfo*, size_t>> plan;
    std::vector<float> norms;
    size_t off = 0;
    te->layers.resize(te->n_layer);
    for (int i = 0; i < te->n_layer; ++i) {
        Layer& L = te->layers[i];
        const std::string p = "blk." + std::to_string(i) + ".";
        TE_TRY(find_weight(te, p + "attn_q.weight", C, Q, L.q, off, plan));
        TE_TRY(find_weight(te, p + "attn_k.weight", C, K, L.k, off, plan));
        TE_TRY(find_weight(te, p + "attn_v.weight", C, K, L.v, off, plan));
        TE_TRY(find_weight(te, p + "attn_output.weight", Q, C, L.o, off, plan));
        TE_TRY(find_weight(te, p + "ffn_gate.weight", C, F, L.gate, off, plan));
        TE_TRY(find_weight(te, p + "ffn_up.weight", C, F, L.up, off, plan));
        TE_TRY(find_weight(te, p + "ffn_down.weight", F, C, L.down, off, plan));
        TE_TRY(find_norm(te, p + "attn_norm.weight", C, L.attn_norm, norms));
        TE_TRY(find_norm(te, p + "ffn_norm.weight", C, L.ffn_norm, norms));
        TE_TRY(find_norm(te, p + "attn_q_norm.weight", HD, L.q_norm, norms));
        TE_TRY(find_norm(te, p + "attn_k_norm.weight", HD, L.k_norm, norms));
    }
    te->weight_bytes = off;
    te->norm_bytes = norms.size() * sizeof(float);
    te->stage_elems = std::max<size_t>({(size_t) (Q + 2 * K) * C, (size_t) 2 * F * C, (size_t) C * Q, (size_t) C * F});

    if (device >= 0) TE_CUDA(cudaSetDevice(device));
    TE_CUDA(cudaGetDevice(&te->device));
    TE_CUDA(cudaStreamCreateWithFlags(&te->stream, cudaStreamNonBlocking));
    TE_BLAS(cublasCreate(&te->blas));
    TE_BLAS(cublasSetStream(te->blas, te->stream));
    TE_CUDA(cudaMalloc((void**) &te->d_w, te->weight_bytes));
    TE_CUDA(cudaMalloc((void**) &te->d_norm, te->norm_bytes));
    TE_CUDA(cudaMalloc((void**) &te->d_stage, te->stage_elems * sizeof(uint16_t)));
    const auto t1 = std::chrono::steady_clock::now();
    for (const auto& [t, o] : plan)
        TE_CUDA(cudaMemcpy(te->d_w + o, f.tensor_data(*t), te_row_bytes(t->type, (int64_t) t->shape[0]) * t->shape[1],
                           cudaMemcpyHostToDevice));
    TE_CUDA(cudaMemcpy(te->d_norm, norms.data(), te->norm_bytes, cudaMemcpyHostToDevice));
    const auto t2 = std::chrono::steady_clock::now();
    te->read_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    te->upload_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    return std::string();
}

// Y[T][N] (f32, ldc = ldy) = X16[T][K] . W16[N][K]^T (+ Y when accumulate)
std::string gemm(Qwen3VlTe* te, const uint16_t* x16, int T, int K, const uint16_t* w16, int N, float* y, int ldy,
                 bool accumulate) {
    const float one = 1.f, beta = accumulate ? 1.f : 0.f;
    TE_BLAS(cublasGemmEx(te->blas, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &one, w16, CUDA_R_16F, K, x16, CUDA_R_16F, K, &beta,
                         y, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    return std::string();
}

std::string decode(Qwen3VlTe* te, const WeightRef& w, uint16_t* dst) {
    return te_dequant_f16(w.type, te->d_w + w.off, w.ne0, w.rows, dst, te->stream);
}

struct Work {   // per-encode device buffers
    std::vector<void*> ptrs;
    size_t bytes = 0;
    ~Work() { for (void* p : ptrs) cudaFree(p); }
    template <class T> std::string get(T** p, size_t n) {
        TE_CUDA(cudaMalloc((void**) p, std::max<size_t>(n, 1) * sizeof(T)));
        ptrs.push_back(*p);
        bytes += std::max<size_t>(n, 1) * sizeof(T);
        return std::string();
    }
};

std::string encode_impl(Qwen3VlTe* te, const int32_t* ids, int T, float* out, Qwen3VlTeStats* st) {
    const int C = te->C, F = te->F, H = te->H, KV = te->KV, Q = H * HD, KD = KV * HD, R = H / KV;
    const int vocab = (int) te->embd->shape[1];
    std::vector<float> emb((size_t) T * C);
    const size_t erb = te_row_bytes(te->embd->type, C);
    for (int t = 0; t < T; ++t) {
        if (ids[t] < 0 || ids[t] >= vocab) return "qwen3vl_te: token id " + std::to_string(ids[t]) + " outside the vocabulary";
        TE_TRY(te_dequant_f32_host(te->embd->type, te->file->tensor_data(*te->embd) + (size_t) ids[t] * erb, C, 1,
                                   emb.data() + (size_t) t * C));
    }
    TE_CUDA(cudaSetDevice(te->device));
    Work wk;
    float *x, *qkv, *att, *gu, *S;
    uint16_t *h16, *q16, *k16, *v16, *a16, *P;
    float2* cs;
    const int qc = std::min(T, QCHUNK);
    TE_TRY(wk.get(&x, (size_t) T * C));
    TE_TRY(wk.get(&h16, (size_t) T * std::max(C, Q)));
    TE_TRY(wk.get(&qkv, (size_t) T * (Q + 2 * KD)));
    TE_TRY(wk.get(&q16, (size_t) T * Q));
    TE_TRY(wk.get(&k16, (size_t) T * KD));
    TE_TRY(wk.get(&v16, (size_t) T * KD));
    TE_TRY(wk.get(&att, (size_t) T * Q));
    TE_TRY(wk.get(&gu, (size_t) T * 2 * F));
    TE_TRY(wk.get(&a16, (size_t) T * F));
    TE_TRY(wk.get(&S, (size_t) KV * qc * T));
    TE_TRY(wk.get(&P, (size_t) KV * qc * T));
    TE_TRY(wk.get(&cs, (size_t) T * (HD / 2)));
    cudaEvent_t e0, e1;
    TE_CUDA(cudaEventCreate(&e0));
    TE_CUDA(cudaEventCreate(&e1));
    TE_CUDA(cudaEventRecord(e0, te->stream));
    TE_CUDA(cudaMemcpyAsync(x, emb.data(), emb.size() * sizeof(float), cudaMemcpyHostToDevice, te->stream));
    rope_table_kernel<<<(T * (HD / 2) + 255) / 256, 256, 0, te->stream>>>(T, te->theta, cs);
    const float scale = 1.0f / sqrtf((float) HD);
    const float one = 1.f, zero = 0.f;
    uint16_t* W = te->d_stage;
    for (int li = 0; li < te->n_layer; ++li) {
        const Layer& L = te->layers[li];
        const float* nw = te->d_norm;
        // attention
        rms_f16_kernel<<<T, 256, 0, te->stream>>>(x, nw + L.attn_norm, C, te->eps, h16);
        TE_TRY(decode(te, L.q, W));
        TE_TRY(decode(te, L.k, W + (size_t) Q * C));
        TE_TRY(decode(te, L.v, W + (size_t) (Q + KD) * C));
        TE_TRY(gemm(te, h16, T, C, W, Q + 2 * KD, qkv, Q + 2 * KD, false));
        qk_rope_kernel<<<dim3(T, H + 2 * KV), HD, 0, te->stream>>>(qkv, T, H, KV, nw + L.q_norm, nw + L.k_norm, te->eps,
                                                                   cs, q16, k16, v16);
        for (int q0 = 0; q0 < T; q0 += QCHUNK) {
            const int n = std::min(QCHUNK, T - q0);
            const int kn = q0 + n;                      // keys 0 .. kn-1 (causal)
            for (int r = 0; r < R; ++r) {
                // S[g][j][c] = q(head g R + r, query q0 + j) . k(group g, key c)
                TE_BLAS(cublasGemmStridedBatchedEx(
                    te->blas, CUBLAS_OP_T, CUBLAS_OP_N, kn, n, HD, &one, k16, CUDA_R_16F, HD, (long long) T * HD,
                    q16 + ((size_t) r * T + q0) * HD, CUDA_R_16F, HD, (long long) R * T * HD, &zero, S, CUDA_R_32F, kn,
                    (long long) n * kn, KV, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                softmax_causal_kernel<<<KV * n, 256, 0, te->stream>>>(S, kn, kn, n, q0, scale, P);
                // O(head g R + r, query q0 + j) = P[g][j] . v(group g): written into att[q0 + j][(g R + r) HD ..]
                TE_BLAS(cublasGemmStridedBatchedEx(
                    te->blas, CUBLAS_OP_N, CUBLAS_OP_N, HD, n, kn, &one, v16, CUDA_R_16F, HD, (long long) T * HD, P,
                    CUDA_R_16F, kn, (long long) n * kn, &zero, att + (size_t) q0 * Q + (size_t) r * HD, CUDA_R_32F, Q,
                    (long long) R * HD, KV, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            }
        }
        cast_f16_kernel<<<grid_for((size_t) T * Q), 256, 0, te->stream>>>(att, (size_t) T * Q, h16);
        TE_TRY(decode(te, L.o, W));
        TE_TRY(gemm(te, h16, T, Q, W, C, x, C, true));
        // feed-forward
        rms_f16_kernel<<<T, 256, 0, te->stream>>>(x, nw + L.ffn_norm, C, te->eps, h16);
        TE_TRY(decode(te, L.gate, W));
        TE_TRY(decode(te, L.up, W + (size_t) F * C));
        TE_TRY(gemm(te, h16, T, C, W, 2 * F, gu, 2 * F, false));
        swiglu_f16_kernel<<<grid_for((size_t) T * F), 256, 0, te->stream>>>(gu, T, F, a16);
        TE_TRY(decode(te, L.down, W));
        TE_TRY(gemm(te, a16, T, F, W, C, x, C, true));
    }
    TE_CUDA(cudaGetLastError());
    TE_CUDA(cudaMemcpyAsync(out, x, (size_t) T * C * sizeof(float), cudaMemcpyDeviceToHost, te->stream));
    TE_CUDA(cudaEventRecord(e1, te->stream));
    TE_CUDA(cudaStreamSynchronize(te->stream));
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    for (size_t i = 0; i < (size_t) T * C; ++i)
        if (!std::isfinite(out[i])) return "qwen3vl_te: the condition has a non-finite value";
    if (st) {
        st->encode_ms = ms;
        st->weight_bytes = te->weight_bytes;
        st->peak_bytes = te->weight_bytes + te->norm_bytes + te->stage_elems * sizeof(uint16_t) + wk.bytes;
    }
    return std::string();
}

}  // namespace

Qwen3VlTe* qwen3vl_te_open(const std::string& gguf_path, int device, std::string* err) {
    auto* te = new Qwen3VlTe();
    std::string e;
    try {
        e = open_impl(te, gguf_path, device);
    } catch (const std::exception& ex) {
        e = std::string("qwen3vl_te: ") + ex.what();
    }
    if (!e.empty()) {
        if (err) *err = e;
        qwen3vl_te_close(te);
        return nullptr;
    }
    return te;
}

int qwen3vl_te_hidden(const Qwen3VlTe* te) { return te ? te->C : 0; }

std::string qwen3vl_te_encode(Qwen3VlTe* te, const int32_t* ids, int n, float* out, Qwen3VlTeStats* stats) {
    if (!te || n <= 0) return "qwen3vl_te: nothing to encode";
    try {
        return encode_impl(te, ids, n, out, stats);
    } catch (const std::exception& ex) {
        return std::string("qwen3vl_te: ") + ex.what();
    }
}

std::string qwen3vl_te_open_report(const Qwen3VlTe* te) {
    char b[160];
    std::snprintf(b, sizeof b, "read %.0f ms, upload %.0f ms (%.2f GB of weights)", te->read_ms, te->upload_ms,
                  te->weight_bytes / 1e9);
    return b;
}

void qwen3vl_te_close(Qwen3VlTe* te) {
    if (!te) return;
    if (te->d_w) cudaFree(te->d_w);
    if (te->d_norm) cudaFree(te->d_norm);
    if (te->d_stage) cudaFree(te->d_stage);
    if (te->blas) cublasDestroy(te->blas);
    if (te->stream) cudaStreamDestroy(te->stream);
    delete te;
}

}  // namespace strata::image
