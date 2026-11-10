// tools/vision/own/vit_sycl.cpp - the VitGpu class (vit_gpu.h) on an Intel GPU: SYCL kernels and oneMKL products (own-stack phase 3, OWN-3V part B).
//
// A line-for-line counterpart of vit_gpu.cu (CUDA / HIP): the same host flow, weights as stored (BF16 / F16; F32 matrices become BF16, Q8_0 becomes F16), the same
// row chunking, the same 16-bit types (F16 q, k, v with 1/sqrt(head size) in q, F16 scores turned into F16 probabilities in place, float32 residual stream,
// 16-bit product inputs with float32 results) and the same diagnostics (STRATA_VIT_REF_ROUND, STRATA_VIT_ROWS, STRATA_VIT_SCORES, STRATA_VIT_PROF).
// Two differences, both forced by oneMKL on the Arc (RESULTS-OWN-3V.md 11):
//   - the P*V product writes the attention context head by head, ctx[h][token][j] (oneMKL refuses batched outputs that interleave), and the o-projection's
//     cast reads it back into the token-major 16-bit input (the same numbers);
//   - the score chunk is larger than on CUDA (oneMKL's batched products with an inner size of 72 are slow on small chunks), sized so the reader's device memory
//     stays at or below mtmd-SYCL's.
// No atomics: every reduction is a work-group reduction of a fixed work-group size, and oneMKL gave one result in eight launches on every shape used here.
#include "vit_gpu.h"

#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata_vit {

namespace {

namespace blas = oneapi::mkl::blas::column_major;
using oneapi::mkl::transpose;
using half = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;

enum Kind { K_F32 = 0, K_F16 = 1, K_BF16 = 2 };   // the 16-bit kind a product's input takes (its weight's)

struct DevW {            // a weight matrix [rows = out][cols = in] on the device
    void* p = nullptr;
    Kind kind = K_BF16;
    int rows = 0, cols = 0;
};

struct DevLayer {
    float *ln1_w = nullptr, *ln1_b = nullptr, *ln2_w = nullptr, *ln2_b = nullptr, *qkv_b = nullptr, *o_b = nullptr, *up_b = nullptr, *down_b = nullptr;
    DevW qkv, o, up, down;
};

inline uint16_t f_to_bf16(float f) {          // round to nearest even
    uint32_t u = sycl::bit_cast<uint32_t>(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40);   // NaN stays NaN
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
inline uint16_t f_to_f16bits(float f) { return sycl::bit_cast<uint16_t>(half(f)); }
inline float f16bits_to_f(uint16_t u) { return (float) sycl::bit_cast<half>(u); }

inline void store_kind(void* out, size_t i, float v, int kind) {
    if (kind == K_F32) ((float*) out)[i] = v;
    else if (kind == K_F16) ((uint16_t*) out)[i] = f_to_f16bits(v);
    else ((uint16_t*) out)[i] = f_to_bf16(v);
}

constexpr int kSoftmaxThreads = 256, kSoftmaxPer = 64;

uint16_t host_f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
uint16_t host_f32_to_f16(float f) {
    const half h(f);
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}

// ---- kernels (the CUDA kernels of vit_gpu.cu; one work-group of `T` items per row unless said otherwise) ----

// x[t] += patch bias + the position table resized (bilinear, align corners) to the token's grid position
void k_patch_post(sycl::queue& q, float* x, const float* bias, const float* pos, int G, int pw, int ph, int nbx, int D, int N) {
    q.parallel_for(sycl::nd_range<1>((size_t) N * 256, 256), [=](sycl::nd_item<1> it) {
        const int t = (int) it.get_group(0), b = t / 4, r = t % 4;
        const int ty = 2 * (b / nbx) + r / 2, tx = 2 * (b % nbx) + r % 2;
        float sfx = (float) pw / G, sfy = (float) ph / G;
        if (pw > 1 && G > 1) sfx = (float) (pw - 1) / (G - 1);
        if (ph > 1 && G > 1) sfy = (float) (ph - 1) / (G - 1);
        const float fy = (float) ty / sfy, fx = (float) tx / sfx;
        int y0 = (int) sycl::floor(fy), x0 = (int) sycl::floor(fx), y1 = y0 + 1, x1 = x0 + 1;
        y0 = sycl::max(0, sycl::min(y0, G - 1)); y1 = sycl::max(0, sycl::min(y1, G - 1));
        x0 = sycl::max(0, sycl::min(x0, G - 1)); x1 = sycl::max(0, sycl::min(x1, G - 1));
        const float dy = sycl::fmax(0.0f, sycl::fmin(fy - (float) y0, 1.0f)), dx = sycl::fmax(0.0f, sycl::fmin(fx - (float) x0, 1.0f));
        const float* a = pos + ((size_t) y0 * G + x0) * D;
        const float* bb = pos + ((size_t) y0 * G + x1) * D;
        const float* c = pos + ((size_t) y1 * G + x0) * D;
        const float* d = pos + ((size_t) y1 * G + x1) * D;
        float* xr = x + (size_t) t * D;
        for (int i = (int) it.get_local_id(0); i < D; i += 256)
            xr[i] += bias[i] + (a[i] * (1 - dx) * (1 - dy) + bb[i] * dx * (1 - dy) + c[i] * (1 - dx) * dy + d[i] * dx * dy);
    });
}

// out = LayerNorm(x) * w + b, into `kind`; rows rows of D
void k_layernorm(sycl::queue& q, const float* x, const float* w, const float* b, float eps, int D, void* out, int kind, int rows) {
    q.parallel_for(sycl::nd_range<1>((size_t) rows * 256, 256), [=](sycl::nd_item<1> it) {
        const size_t row = it.get_group(0);
        const int lid = (int) it.get_local_id(0);
        const float* xr = x + row * D;
        float s = 0.0f;
        for (int i = lid; i < D; i += 256) s += xr[i];
        const float mean = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / (float) D;
        float v = 0.0f;
        for (int i = lid; i < D; i += 256) { const float d = xr[i] - mean; v += d * d; }
        const float scale = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>()) / (float) D + eps);
        for (int i = lid; i < D; i += 256) store_kind(out, row * D + i, (xr[i] - mean) * scale * w[i] + b[i], kind);
    });
}

// y [rows][3D] (the fused qkv product of the tokens row0 ..) + bias -> q, k (rotary applied), v, each [N][D] in F16
void k_qkv_rope(sycl::queue& q, const float* y, const float* bias, int row0, int nbx, int heads, int hd, float theta_scale, float qscale, uint16_t* qo, uint16_t* ko,
                uint16_t* vo, int rows) {
    q.parallel_for(sycl::nd_range<1>((size_t) rows * 256, 256), [=](sycl::nd_item<1> it) {
        const int t = row0 + (int) it.get_group(0), D = heads * hd, half_ = hd / 2, quarter = hd / 4;
        const int b = t / 4, r = t % 4;
        const int ty = 2 * (b / nbx) + r / 2, tx = 2 * (b % nbx) + r % 2;
        const float* yr = y + it.get_group(0) * 3 * (size_t) D;
        const int lid = (int) it.get_local_id(0);
        for (int i = lid; i < 2 * heads * half_; i += 256) {
            const int part = i / (heads * half_), rem = i % (heads * half_), h = rem / half_, j = rem % half_;
            const int base = part * D + h * hd;
            const float a = yr[base + j] + bias[base + j], c = yr[base + j + half_] + bias[base + j + half_];
            const float freq = sycl::pow(theta_scale, (float) (j % quarter));
            const float ang = (float) (j < quarter ? ty : tx) * freq;
            const float sn = sycl::sin(ang), cs = sycl::cos(ang);
            uint16_t* dst = (part == 0 ? qo : ko) + (size_t) t * D + h * hd;
            const float sc = part == 0 ? qscale : 1.0f;
            dst[j] = f_to_f16bits((a * cs - c * sn) * sc);
            dst[j + half_] = f_to_f16bits((a * sn + c * cs) * sc);
        }
        for (int e = lid; e < D; e += 256) vo[(size_t) t * D + e] = f_to_f16bits(yr[2 * D + e] + bias[2 * D + e]);
    });
}

// In place: rows of n F16 scores become F16 probabilities.  T items and PER scores per item (T * PER >= n), the row held in registers.
template <int T, int PER>
void k_softmax_rows(sycl::queue& q, uint16_t* S, int n, int ld, int rows) {
    q.parallel_for(sycl::nd_range<1>((size_t) rows * T, T), [=](sycl::nd_item<1> it) {
        uint16_t* s = S + it.get_group(0) * (size_t) ld;
        const int lid = (int) it.get_local_id(0);
        float v[PER];
        float m = -FLT_MAX;
#pragma unroll
        for (int k = 0; k < PER; ++k) {
            const int j = lid + k * T;
            v[k] = j < n ? f16bits_to_f(s[j]) : -FLT_MAX;
            m = sycl::fmax(m, v[k]);
        }
        m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < PER; ++k) {
            const int j = lid + k * T;
            v[k] = j < n ? sycl::exp(v[k] - m) : 0.0f;
            sum += v[k];
        }
        sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
        const float inv = 1.0f / sum;
#pragma unroll
        for (int k = 0; k < PER; ++k) {
            const int j = lid + k * T;
            if (j < n) s[j] = f_to_f16bits(v[k] * inv);
        }
    });
}

void softmax_rows(sycl::queue& q, uint16_t* S, int n, int ld, int rows) {
    if (n <= 128) k_softmax_rows<64, 2>(q, S, n, ld, rows);
    else if (n <= 512) k_softmax_rows<64, 8>(q, S, n, ld, rows);
    else if (n <= 1024) k_softmax_rows<128, 8>(q, S, n, ld, rows);
    else if (n <= 4096) k_softmax_rows<256, 16>(q, S, n, ld, rows);
    else k_softmax_rows<256, 64>(q, S, n, ld, rows);
}

// a row too long for registers (more than 16,384 patches): three passes over the row in memory
void k_softmax_rows_long(sycl::queue& q, uint16_t* S, int n, int ld, int rows) {
    q.parallel_for(sycl::nd_range<1>((size_t) rows * kSoftmaxThreads, kSoftmaxThreads), [=](sycl::nd_item<1> it) {
        uint16_t* s = S + it.get_group(0) * (size_t) ld;
        const int lid = (int) it.get_local_id(0);
        float m = -FLT_MAX;
        for (int j = lid; j < n; j += kSoftmaxThreads) m = sycl::fmax(m, f16bits_to_f(s[j]));
        m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
        float sum = 0.0f;
        for (int j = lid; j < n; j += kSoftmaxThreads) sum += sycl::exp(f16bits_to_f(s[j]) - m);
        sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
        const float inv = 1.0f / sum;
        for (int j = lid; j < n; j += kSoftmaxThreads) s[j] = f_to_f16bits(sycl::exp(f16bits_to_f(s[j]) - m) * inv);
    });
}

size_t grid_for(size_t n) { return std::min<size_t>((n + 255) / 256, 65535) * 256; }

// y[i] rounded to the 16-bit kind and back (STRATA_VIT_REF_ROUND)
void k_round(sycl::queue& q, float* y, size_t n, int kind) {
    const size_t g = grid_for(n);
    q.parallel_for(sycl::nd_range<1>(g, 256), [=](sycl::nd_item<1> it) {
        for (size_t i = it.get_global_id(0); i < n; i += g) {
            const float v = y[i];
            y[i] = kind == K_F16 ? (float) half(v) : sycl::bit_cast<float>((uint32_t) f_to_bf16(v) << 16);
        }
    });
}

// the head-major attention context ctx[h][token][j] (N tokens) of rows r0 .. r0 + rows to the token-major 16-bit input out[r][h * hd + j]
void k_cast_ctx(sycl::queue& q, const float* ctx, int N, int heads, int hd, int r0, int rows, void* out, int kind) {
    const int D = heads * hd;
    q.parallel_for(sycl::nd_range<1>((size_t) rows * 256, 256), [=](sycl::nd_item<1> it) {
        const int r = (int) it.get_group(0);
        for (int e = (int) it.get_local_id(0); e < D; e += 256) {
            const int h = e / hd, j = e % hd;
            store_kind(out, (size_t) r * D + e, ctx[((size_t) h * N + (size_t) (r0 + r)) * hd + j], kind);
        }
    });
}

// x[r] += y[r] + bias
void k_add_bias_res(sycl::queue& q, float* x, const float* y, const float* bias, int D, int rows) {
    q.parallel_for(sycl::nd_range<1>((size_t) rows * 256, 256), [=](sycl::nd_item<1> it) {
        const size_t base = it.get_group(0) * (size_t) D;
        for (int i = (int) it.get_local_id(0); i < D; i += 256) x[base + i] += y[base + i] + bias[i];
    });
}

// out = gelu_tanh(y + bias) into `kind`
void k_bias_gelu_cast(sycl::queue& q, const float* y, const float* bias, int n, void* out, int kind, int rows) {
    q.parallel_for(sycl::nd_range<1>((size_t) rows * 256, 256), [=](sycl::nd_item<1> it) {
        const size_t base = it.get_group(0) * (size_t) n;
        for (int i = (int) it.get_local_id(0); i < n; i += 256) {
            const float x = y[base + i] + bias[i];
            const float g = 0.5f * x * (1.0f + sycl::tanh(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x)));
            store_kind(out, base + i, g, kind);
        }
    });
}

// y[r] += bias
void k_add_bias(sycl::queue& q, float* y, const float* bias, int n, int rows) {
    q.parallel_for(sycl::nd_range<1>((size_t) rows * 256, 256), [=](sycl::nd_item<1> it) {
        const size_t base = it.get_group(0) * (size_t) n;
        for (int i = (int) it.get_local_id(0); i < n; i += 256) y[base + i] += bias[i];
    });
}

// the GPU this reader uses: the Level Zero (else any) GPU with the most memory; ONEAPI_DEVICE_SELECTOR filters the list
bool pick_gpu(sycl::device& out, std::string& why) {
    try {
        std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
        if (gpus.empty()) { why = "no GPU found"; return false; }
        auto score = [](const sycl::device& d) {
            const bool l0 = d.get_backend() == sycl::backend::ext_oneapi_level_zero;
            return std::make_pair(l0 ? 1 : 0, (unsigned long long) d.get_info<sycl::info::device::global_mem_size>());
        };
        out = *std::max_element(gpus.begin(), gpus.end(), [&](const sycl::device& a, const sycl::device& b) { return score(a) < score(b); });
        if (!out.has(sycl::aspect::fp16)) { why = "the GPU has no 16-bit float support"; return false; }
        return true;
    } catch (const sycl::exception& e) {
        why = std::string("no GPU runtime: ") + e.what();
        return false;
    }
}

}  // namespace

struct VitGpuImpl {
    VitGeometry g;
    sycl::queue q;
    std::string async_err;              // the first asynchronous error the queue reported
    std::vector<void*> allocs;          // every device allocation (weights and work buffers), freed in the destructor
    size_t bytes = 0, work_bytes = 0;
    size_t n_weight_allocs = 0;

    float *patch_w = nullptr, *patch_b = nullptr, *pos = nullptr, *post_w = nullptr, *post_b = nullptr, *mm0_b = nullptr, *mm2_b = nullptr;
    DevW mm0, mm2;
    std::vector<DevLayer> layers;

    int cap_patches = 0;
    float *d_x = nullptr, *d_ctx = nullptr, *d_y = nullptr, *d_out = nullptr;   // as vit_gpu.cu; d_ctx holds the context HEAD-MAJOR [H][N][HD]
    uint16_t *d_a = nullptr, *d_f16 = nullptr;
    uint16_t *d_q16 = nullptr, *d_k16 = nullptr, *d_v16 = nullptr;
    uint16_t* d_S = nullptr;
    size_t sp_elems = 0;
    int kRows = 512;
    // scores per attention chunk: 32M elements = 64 MB of F16 (CUDA: 8M).  oneMKL's batched products with an inner size of 72 need wide chunks to be fast
    // on the Arc (RESULTS-OWN-3V.md 11.1); 64 MB keeps the reader at or below mtmd-SYCL's memory at 1,024 tokens.  STRATA_VIT_SCORES (M elements) overrides.
    size_t kScoreElems = (size_t) 32 << 20;
    bool round_out = false;

    ~VitGpuImpl() {
        try {
            q.wait();
        } catch (...) {
        }
        for (void* p : allocs) sycl::free(p, q);
    }
    template <class T> bool alloc(T*& p, size_t n, std::string& err, bool weight) {
        void* m = nullptr;
        try {
            m = sycl::malloc_device(n * sizeof(T), q);
        } catch (const sycl::exception& e) {
            err = std::string("GPU memory: ") + e.what();
            return false;
        }
        if (!m) { err = "GPU memory: out of device memory"; return false; }
        p = (T*) m;
        allocs.push_back(m);
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
        q.memcpy(dst, h.data(), h.size() * 4).wait();
        return true;
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
        q.memcpy(w.p, raw.data(), n * 2).wait();
        return true;
    }

    // y[M][w.rows] (float32) = A[M][w.cols] (kind of w) * w^T
    void gemm(const DevW& w, const void* A, int M, float* y) {
        const int m = w.rows, k = w.cols;
        if (w.kind == K_F32)
            blas::gemm(q, transpose::trans, transpose::nontrans, m, M, k, 1.0f, (const float*) w.p, k, (const float*) A, k, 0.0f, y, m);
        else if (w.kind == K_F16)
            blas::gemm(q, transpose::trans, transpose::nontrans, m, M, k, 1.0f, (const half*) w.p, k, (const half*) A, k, 0.0f, y, m);
        else
            blas::gemm(q, transpose::trans, transpose::nontrans, m, M, k, 1.0f, (const bf16*) w.p, k, (const bf16*) A, k, 0.0f, y, m);
        if (round_out && w.kind != K_F32) k_round(q, y, (size_t) M * w.rows, (int) w.kind);
    }

    bool size_buffers(int patches, std::string& err) {
        if (patches <= cap_patches) return true;
        q.wait();
        for (size_t i = allocs.size(); i > n_weight_allocs; --i) sycl::free(allocs[i - 1], q);
        bytes -= work_bytes;
        work_bytes = 0;
        allocs.resize(n_weight_allocs);
        const size_t N = (size_t) patches, D = (size_t) g.width, F = (size_t) std::max(g.ffn, g.merger_hidden);
        const size_t K0 = (size_t) 3 * g.patch * g.patch;
        d_x = d_y = d_ctx = d_out = nullptr;
        d_a = d_f16 = d_q16 = d_k16 = d_v16 = nullptr;
        d_S = nullptr;
        const size_t Np = (N + 7) / 8 * 8;
        sp_elems = std::min<size_t>((size_t) g.heads * N * Np, std::max<size_t>(kScoreElems, (size_t) g.heads * Np * 64));
        const size_t R = std::min<size_t>(N, (size_t) kRows);
        const size_t ctx_elems = std::max({N * D, N * K0, (N / 4) * (size_t) g.out_dim});
        bool ok = alloc(d_x, N * D, err, false) && alloc(d_ctx, ctx_elems, err, false) && alloc(d_y, R * std::max<size_t>(3 * D, F), err, false) &&
                  alloc(d_a, R * D, err, false) && alloc(d_f16, R * F, err, false) &&
                  alloc(d_q16, N * D, err, false) && alloc(d_k16, N * D, err, false) && alloc(d_v16, N * D, err, false) &&
                  alloc(d_S, sp_elems, err, false);
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
    sycl::device d;
    return pick_gpu(d, why);
}

bool VitGpu::prefer_mtmd(std::string& why) {
    (void) why;
    return false;
}

bool VitGpu::load(const Gguf& gg, const VitGeometry& g, int device, int max_patches, std::string& err) {
    VitGpuImpl& m = *impl_;
    (void) device;          // one GPU: the one pick_gpu chooses
    std::string why;
    if (!vit_geometry_supported(g, why)) { err = "this tower is not supported: " + why; return false; }
    m.g = g;
    sycl::device dev;
    if (!pick_gpu(dev, why)) { err = "cannot use the GPU: " + why; return false; }
    try {
        std::string* sink = &m.async_err;
        m.q = sycl::queue(dev, [sink](sycl::exception_list l) {
            for (auto& e : l) {
                try {
                    std::rethrow_exception(e);
                } catch (const std::exception& x) {
                    if (sink->empty()) *sink = x.what();
                }
            }
        }, sycl::property::queue::in_order());
        {   // the patch embedding: the two temporal taps summed, kept in float32 (a small matrix)
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
            m.q.memcpy(m.patch_w, w0.data(), w0.size() * 4).wait();
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
        if (const char* e = std::getenv("STRATA_VIT_REF_ROUND")) m.round_out = std::atoi(e) != 0;
        if (const char* e = std::getenv("STRATA_VIT_ROWS")) { const int v = std::atoi(e); if (v >= 16) m.kRows = v; }
        if (const char* e = std::getenv("STRATA_VIT_SCORES")) { const int v = std::atoi(e); if (v >= 1) m.kScoreElems = (size_t) v << 20; }
        return m.size_buffers(std::max(max_patches, 16), err);
    } catch (const std::exception& e) {
        err = std::string("the GPU failed while loading the vision encoder: ") + e.what();
        return false;
    }
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
    sycl::queue& q = m.q;
    try {
        if (!m.size_buffers(N, err)) return false;
        static const bool prof = std::getenv("STRATA_VIT_PROF") != nullptr;
        double pt[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        auto pclock = [] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
        double pmark = prof ? pclock() : 0.0;
        auto stage = [&](int i) {
            if (!prof) return;
            q.wait();
            const double now = pclock();
            pt[i] += now - pmark;
            pmark = now;
        };
        auto sync_err = [&](const char* what) {
            q.wait_and_throw();
            if (!m.async_err.empty()) { err = std::string(what) + ": " + m.async_err; m.async_err.clear(); return false; }
            return true;
        };
        const int K0 = 3 * P * P;
        {   // the patch matrix [N][K0], channel-major per patch, tokens in 2x2-merge order, built on the host; it lives in the context buffer
            std::vector<float> pm((size_t) N * K0);
            const size_t plane = (size_t) img.w * img.h;
            for (int t = 0; t < N; ++t) {
                const int b = t / 4, r = t % 4, py = 2 * (b / nbx) + r / 2, px = 2 * (b % nbx) + r % 2;
                float* a = &pm[(size_t) t * K0];
                for (int c = 0; c < 3; ++c)
                    for (int ky = 0; ky < P; ++ky)
                        std::memcpy(a + (size_t) c * P * P + (size_t) ky * P, &img.chw[c * plane + (size_t) (py * P + ky) * img.w + (size_t) px * P], (size_t) P * sizeof(float));
            }
            q.memcpy(m.d_ctx, pm.data(), pm.size() * 4).wait();
        }
        stage(0);
        DevW pw_w;
        pw_w.p = m.patch_w; pw_w.kind = K_F32; pw_w.rows = D; pw_w.cols = K0;
        m.gemm(pw_w, m.d_ctx, N, m.d_x);
        k_patch_post(q, m.d_x, m.patch_b, m.pos, g.pos_grid, pw, ph, nbx, D, N);
        if (!sync_err("patch embedding")) return false;
        stage(1);

        const float theta_scale = std::pow(g.rope_theta, -2.0f / (float) (HD / 2));
        const float kq_scale = 1.0f / std::sqrt((float) HD);
        const int R = m.kRows;
        uint16_t* const a16 = m.d_a;
        for (const DevLayer& L : m.layers) {
            for (int r0 = 0; r0 < N; r0 += R) {
                const int rows_ = std::min(R, N - r0);
                k_layernorm(q, m.d_x + (size_t) r0 * D, L.ln1_w, L.ln1_b, g.eps, D, a16, (int) L.qkv.kind, rows_);
                m.gemm(L.qkv, a16, rows_, m.d_y);
                k_qkv_rope(q, m.d_y, L.qkv_b, r0, nbx, H, HD, theta_scale, kq_scale, m.d_q16, m.d_k16, m.d_v16, rows_);
            }
            stage(2);
            // per chunk of queries: S = K Q^T (one batched product over the heads), softmax, O = V P written head-major (ctx[h][token][j])
            const int Np = (N + 7) / 8 * 8;
            const int QC = (int) std::max<size_t>(1, std::min<size_t>((size_t) N, m.sp_elems / ((size_t) H * Np)));
            for (int q0 = 0; q0 < N; q0 += QC) {
                const int qc = std::min(QC, N - q0);
                blas::gemm_batch(q, transpose::trans, transpose::nontrans, N, qc, HD, half(1.0f), (const half*) m.d_k16, D, HD, (const half*) (m.d_q16 + (size_t) q0 * D), D,
                                 HD, half(0.0f), (half*) m.d_S, Np, (int64_t) qc * Np, H);
                stage(8);
                if (N <= kSoftmaxThreads * kSoftmaxPer) softmax_rows(q, m.d_S, N, Np, H * qc);
                else k_softmax_rows_long(q, m.d_S, N, Np, H * qc);
                stage(9);
                blas::gemm_batch(q, transpose::nontrans, transpose::nontrans, HD, qc, N, 1.0f, (const half*) m.d_v16, D, HD, (const half*) m.d_S, Np, (int64_t) qc * Np, 0.0f,
                                 m.d_ctx + (size_t) q0 * HD, HD, (int64_t) N * HD, H);
                stage(10);
            }
            stage(3);
            for (int r0 = 0; r0 < N; r0 += R) {      // the output projection in chunks: the head-major context rows to the 16-bit input, the product, the residual
                const int rows_ = std::min(R, N - r0);
                k_cast_ctx(q, m.d_ctx, N, H, HD, r0, rows_, a16, (int) L.o.kind);
                m.gemm(L.o, a16, rows_, m.d_y);
                k_add_bias_res(q, m.d_x + (size_t) r0 * D, m.d_y, L.o_b, D, rows_);
            }
            stage(4);
            for (int r0 = 0; r0 < N; r0 += R) {      // the feed-forward block in chunks
                const int rows_ = std::min(R, N - r0);
                k_layernorm(q, m.d_x + (size_t) r0 * D, L.ln2_w, L.ln2_b, g.eps, D, a16, (int) L.up.kind, rows_);
                m.gemm(L.up, a16, rows_, m.d_y);
                k_bias_gelu_cast(q, m.d_y, L.up_b, F, m.d_f16, (int) L.down.kind, rows_);
                m.gemm(L.down, m.d_f16, rows_, m.d_y);
                k_add_bias_res(q, m.d_x + (size_t) r0 * D, m.d_y, L.down_b, D, rows_);
            }
            stage(5);
        }
        if (!sync_err("the layers")) return false;
        const int M = N / 4, RM = std::max(1, R / 4);
        for (int r0 = 0; r0 < M; r0 += RM) {
            const int rows_ = std::min(RM, M - r0);
            k_layernorm(q, m.d_x + (size_t) r0 * 4 * D, m.post_w, m.post_b, g.eps, D, a16, (int) m.mm0.kind, rows_ * 4);
            m.gemm(m.mm0, a16, rows_, m.d_y);
            k_bias_gelu_cast(q, m.d_y, m.mm0_b, g.merger_hidden, m.d_f16, (int) m.mm2.kind, rows_);
            m.gemm(m.mm2, m.d_f16, rows_, m.d_out + (size_t) r0 * g.out_dim);
            k_add_bias(q, m.d_out + (size_t) r0 * g.out_dim, m.mm2_b, g.out_dim, rows_);
        }
        if (!sync_err("the projector")) return false;
        stage(6);
        rows.resize((size_t) M * g.out_dim);
        q.memcpy(rows.data(), m.d_out, rows.size() * 4).wait();
        stage(7);
        if (prof)
            std::fprintf(stderr, "vit-prof N=%d: patch-matrix %.2f, patch-embed %.2f, ln+qkv %.2f, attention %.2f (S product %.2f, softmax %.2f, PV product %.2f), o-proj %.2f, ffn %.2f, merger %.2f, copy-out %.2f ms\n",
                         N, pt[0], pt[1], pt[2], pt[3] + pt[8] + pt[9] + pt[10], pt[8], pt[9], pt[10], pt[4], pt[5], pt[6], pt[7]);
        return true;
    } catch (const std::exception& e) {
        err = std::string("the GPU failed: ") + e.what();
        return false;
    }
}

}  // namespace strata_vit
