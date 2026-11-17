// src/image/dit/qi21_dit.cu - see include/strata/image/qi21_dit.hpp.
//
// One step, as the native runner (tools/image/sdcpp_native/qwen_image_2_1_native.hpp) with the D1+D2 GEMMs computes it:
//   prelude  emb = sinusoid([t, 0], 256); te = silu(linear_2(silu(linear_1(emb)))); mod = modulation.1(te) (row 0 for
//            the image tokens, row 1 = t 0 for the text); scale = norm_out.linear(te row 0)
//   prefix   (once per condition) x = out_layer(gelu(in_layer(RMSNorm(cond) (w + 1)))); 32 blocks over the text rows,
//            causal attention with f32 queries (attn_exact); each block's K / V (f16) kept in the prefix cache
//   image    x = img_in(latent tokens); 32 blocks over the image rows, keys = [the block's cached text K / V; these
//            tokens' K / V], f16 queries (attn_fast; attn_exact where the card has no fast kernel)
//   block    h = LN(x)(1 + m0); qkv = h W^T (one GEMM over q | k | v); per-head RMSNorm + RoPE; attention;
//            x += to_out(att) tanh(m1); h = LN(x)(1 + m2); gu = h Wgu^T; a = up silu(gate); x += out(a) tanh(m3)
//   head     LN(x)(1 + scale) -> proj_out
#include "strata/image/qi21_dit.hpp"

#include "qi21_kernels.hpp"
#include "strata/image/dit_gemm.hpp"
#include "strata/image/qi21_rope.hpp"
#include "strata/image/te_dequant.hpp"

#define STRATA_GGUF_MAIN_DISABLED 1
#include "strata/artifact/gguf_reader.hpp"

#if defined(__HIP__) || defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include "strata/hip_compat/cuda_runtime.h"
#else
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

namespace strata::image {
namespace {

constexpr int HD = 128;
constexpr float EPS = 1e-6f;
const std::string PFX = "model.diffusion_model.";

#define DIT_CUDA(x)                                                                                        \
    do {                                                                                                   \
        cudaError_t e_ = (x);                                                                              \
        if (e_ != cudaSuccess) return std::string("qi21_dit: ") + #x + ": " + cudaGetErrorString(e_);      \
    } while (0)
#define DIT_TRY(x)                                                                                         \
    do {                                                                                                   \
        std::string r_ = (x);                                                                              \
        if (!r_.empty()) return r_;                                                                        \
    } while (0)
#define DIT_K(x)                                                                                           \
    do {                                                                                                   \
        int r_ = (x);                                                                                      \
        if (r_ != 0) return std::string("qi21_dit: kernel ") + #x + " returned " + std::to_string(r_);     \
    } while (0)

struct W2 {                     // a 2-D weight on the device: rows of K values, N rows, raw blocks of `type`
    uint32_t type = 0;
    size_t off = 0;
    int64_t K = 0, N = 0;
};

struct Block {
    W2 qkv;                     // q | k | v rows one after the other (N = 3C) when fused, else q only
    W2 k, v;                    // used when not fused
    bool fused = false;
    W2 out, gate_up, down;
    size_t nq = 0, nk = 0;      // float offsets of norm_q / norm_k in the norm buffer
};

}  // namespace

struct Qi21Dit {
    std::unique_ptr<strata::GgufFile> file;
    int device = 0;
    int C = 0, F = 0, H = 0, in_ch = 0, out_ch = 0, ctx_dim = 0, t_dim = 0, n_layer = 0;
    bool fast_attn = false;
    std::vector<Block> blocks;
    W2 img_in, txt_in1, txt_in2, lin1, lin2, modw, norm_out, proj_out;
    size_t text_norm = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t ev[3] = {nullptr, nullptr, nullptr};
    uint8_t* d_w = nullptr;
    float* d_norm = nullptr;
    size_t weight_bytes = 0, norm_bytes = 0;
    double open_ms = 0;
    std::vector<std::string> unused;
    // work, sized for t_cap image tokens and l_cap text tokens
    int64_t t_cap = 0, l_cap = 0;
    float *x = nullptr, *h = nullptr, *gu = nullptr, *q16 = nullptr;
    uint16_t *h16 = nullptr, *a16 = nullptr, *m16 = nullptr;   // GEMM inputs, written f16 by the kernels before them
    bool f16_resident = false;                                // weights decoded to f16 once at open (measurement)
    double decode_ms = 0;
    uint16_t *kw = nullptr, *vw = nullptr, *pk = nullptr, *pv = nullptr;
    float *pe_img = nullptr, *pe_txt = nullptr, *latent = nullptr, *outp = nullptr, *ctx_in = nullptr;
    float *tin = nullptr, *emb = nullptr, *te1 = nullptr, *te2 = nullptr, *mod = nullptr, *scale = nullptr;
    size_t work_bytes = 0, peak_bytes = 0;
    std::vector<void*> work;
    // the prefix cache
    uint64_t prefix_key = 0;
    int64_t prefix_len = -1;
};

namespace {

std::string find2(Qi21Dit* d, const std::string& name, int64_t K, int64_t N, W2& w, size_t& off, bool align,
                  std::vector<std::pair<const strata::TensorInfo*, size_t>>& plan, std::set<std::string>& used) {
    const strata::TensorInfo* t = d->file->find(PFX + name);
    if (!t) return "qi21_dit: tensor " + name + " missing";
    if (t->shape.size() != 2 || (int64_t) t->shape[0] != K || (int64_t) t->shape[1] != N)
        return "qi21_dit: tensor " + name + " has an unexpected shape";
    const size_t rb = te_row_bytes(t->type, K);
    if (rb == 0 || !te_type_supported(t->type))
        return "qi21_dit: tensor " + name + " has type " + t->type_name() + ", which the DiT does not read";
    w = {t->type, off, K, N};
    plan.emplace_back(t, off);
    used.insert(PFX + name);
    off += rb * (size_t) N;
    if (align) off = (off + 255) / 256 * 256;
    return std::string();
}

std::string find1(Qi21Dit* d, const std::string& name, int64_t n, size_t& at, std::vector<float>& host,
                  std::set<std::string>& used) {
    const strata::TensorInfo* t = d->file->find(PFX + name);
    if (!t) return "qi21_dit: tensor " + name + " missing";
    if ((int64_t) t->elements() != n) return "qi21_dit: tensor " + name + " has an unexpected size";
    at = host.size();
    host.resize(at + (size_t) n);
    used.insert(PFX + name);
    return te_dequant_f32_host(t->type, d->file->tensor_data(*t), n, 1, host.data() + at);
}

std::string shape_of(Qi21Dit* d, const std::string& name, int64_t& ne0, int64_t& ne1) {
    const strata::TensorInfo* t = d->file->find(PFX + name);
    if (!t || t->shape.size() != 2) return "qi21_dit: tensor " + name + " missing or not 2-D";
    ne0 = (int64_t) t->shape[0];
    ne1 = (int64_t) t->shape[1];
    return std::string();
}

// host -> device through a pinned chunk pair, async on the stream
std::string upload(Qi21Dit* d, uint8_t* dst, const uint8_t* src, size_t n, uint8_t* pinned[2], cudaEvent_t pin_ev[2],
                   size_t chunk, int& next) {
    for (size_t done = 0; done < n;) {
        const int b = next;
        next ^= 1;
        DIT_CUDA(cudaEventSynchronize(pin_ev[b]));
        const size_t c = std::min(chunk, n - done);
        std::memcpy(pinned[b], src + done, c);
        DIT_CUDA(cudaMemcpyAsync(dst + done, pinned[b], c, cudaMemcpyHostToDevice, d->stream));
        DIT_CUDA(cudaEventRecord(pin_ev[b], d->stream));
        done += c;
    }
    return std::string();
}

// QI21_DIT_F16_RESIDENT (a measurement for cards with room, not a memory plan): every 2-D weight decoded once to f16
// into its own buffer (the same f16 values dit_gemm decodes on every call), the raw blocks freed; the GEMMs then
// read the f16 weights in place.
std::string to_f16_resident(Qi21Dit* d) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<W2*> ws;
    for (Block& B : d->blocks) {
        ws.push_back(&B.qkv);
        if (!B.fused) {
            ws.push_back(&B.k);
            ws.push_back(&B.v);
        }
        ws.push_back(&B.out);
        ws.push_back(&B.gate_up);
        ws.push_back(&B.down);
    }
    for (W2* w : {&d->img_in, &d->txt_in1, &d->txt_in2, &d->lin1, &d->lin2, &d->modw, &d->norm_out, &d->proj_out})
        ws.push_back(w);
    size_t total = 0;
    for (const W2* w : ws) total += ((size_t) w->K * (size_t) w->N * sizeof(uint16_t) + 255) / 256 * 256;
    uint8_t* f16 = nullptr;
    DIT_CUDA(cudaMalloc((void**) &f16, total));
    size_t off = 0;
    for (W2* w : ws) {
        const std::string e = te_dequant_f16(w->type, d->d_w + w->off, w->K, w->N,
                                             reinterpret_cast<uint16_t*>(f16 + off), d->stream);
        if (!e.empty()) {
            cudaFree(f16);
            return "qi21_dit: " + e;
        }
        w->type = TE_F16;
        w->off = off;
        off += ((size_t) w->K * (size_t) w->N * sizeof(uint16_t) + 255) / 256 * 256;
    }
    DIT_CUDA(cudaStreamSynchronize(d->stream));
    DIT_CUDA(cudaFree(d->d_w));
    d->d_w = f16;
    d->weight_bytes = total;
    d->decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return std::string();
}

std::string open_impl(Qi21Dit* d, const std::string& path, int device) {
    const auto t0 = std::chrono::steady_clock::now();
    d->file = std::make_unique<strata::GgufFile>(path);
    std::set<std::string> used;
    int64_t a = 0, b = 0;
    DIT_TRY(shape_of(d, "img_in.weight", a, b));
    d->in_ch = (int) a;
    d->C = (int) b;
    DIT_TRY(shape_of(d, "proj_out.weight", a, b));
    d->out_ch = (int) b;
    DIT_TRY(shape_of(d, "txt_in.in_layer.weight", a, b));
    d->ctx_dim = (int) a;
    DIT_TRY(shape_of(d, "time_text_embed.timestep_embedder.linear_1.weight", a, b));
    d->t_dim = (int) a;
    DIT_TRY(shape_of(d, "transformer_blocks.0.img_mlp.gate_up.weight", a, b));
    d->F = (int) (b / 2);
    const int C = d->C, F = d->F;
    if (C % HD != 0 || (C / HD) % 8 != 0 || C > 8192) return "qi21_dit: unexpected model width";
    d->H = C / HD;
    while (d->file->find(PFX + "transformer_blocks." + std::to_string(d->n_layer) + ".attn.to_out.0.weight")) ++d->n_layer;
    if (d->n_layer == 0) return "qi21_dit: no transformer blocks";

    std::vector<std::pair<const strata::TensorInfo*, size_t>> plan;
    std::vector<float> norms;
    size_t off = 0;
    d->blocks.resize(d->n_layer);
    for (int i = 0; i < d->n_layer; ++i) {
        Block& B = d->blocks[i];
        const std::string p = "transformer_blocks." + std::to_string(i) + ".";
        const strata::TensorInfo* tq = d->file->find(PFX + p + "attn.to_q.weight");
        const strata::TensorInfo* tk = d->file->find(PFX + p + "attn.to_k.weight");
        const strata::TensorInfo* tv = d->file->find(PFX + p + "attn.to_v.weight");
        B.fused = tq && tk && tv && tq->type == tk->type && tq->type == tv->type;
        DIT_TRY(find2(d, p + "attn.to_q.weight", C, C, B.qkv, off, !B.fused, plan, used));
        DIT_TRY(find2(d, p + "attn.to_k.weight", C, C, B.k, off, !B.fused, plan, used));
        DIT_TRY(find2(d, p + "attn.to_v.weight", C, C, B.v, off, true, plan, used));
        if (B.fused) B.qkv.N = 3 * C;   // the three tensors' rows one after the other: one weight of 3C rows
        DIT_TRY(find2(d, p + "attn.to_out.0.weight", C, C, B.out, off, true, plan, used));
        DIT_TRY(find2(d, p + "img_mlp.gate_up.weight", C, 2 * F, B.gate_up, off, true, plan, used));
        DIT_TRY(find2(d, p + "img_mlp.out.weight", F, C, B.down, off, true, plan, used));
        DIT_TRY(find1(d, p + "attn.norm_q.weight", HD, B.nq, norms, used));
        DIT_TRY(find1(d, p + "attn.norm_k.weight", HD, B.nk, norms, used));
    }
    DIT_TRY(find2(d, "img_in.weight", d->in_ch, C, d->img_in, off, true, plan, used));
    DIT_TRY(find2(d, "txt_in.in_layer.weight", d->ctx_dim, C, d->txt_in1, off, true, plan, used));
    DIT_TRY(find2(d, "txt_in.out_layer.weight", C, C, d->txt_in2, off, true, plan, used));
    DIT_TRY(find2(d, "time_text_embed.timestep_embedder.linear_1.weight", d->t_dim, C, d->lin1, off, true, plan, used));
    DIT_TRY(find2(d, "time_text_embed.timestep_embedder.linear_2.weight", C, C, d->lin2, off, true, plan, used));
    DIT_TRY(find2(d, "modulation.1.weight", C, 4 * C, d->modw, off, true, plan, used));
    DIT_TRY(find2(d, "norm_out.linear.weight", C, C, d->norm_out, off, true, plan, used));
    DIT_TRY(find2(d, "proj_out.weight", C, d->out_ch, d->proj_out, off, true, plan, used));
    DIT_TRY(find1(d, "txt_in.text_norm.weight", d->ctx_dim, d->text_norm, norms, used));
    for (const auto& t : d->file->tensors())
        if (t.name.compare(0, PFX.size(), PFX) == 0 && !used.count(t.name)) d->unused.push_back(t.name);
    d->weight_bytes = off;
    d->norm_bytes = norms.size() * sizeof(float);

    if (device >= 0) DIT_CUDA(cudaSetDevice(device));
    DIT_CUDA(cudaGetDevice(&d->device));
    d->fast_attn = qi21::has_fast_attn(d->device) != 0;
    DIT_CUDA(cudaStreamCreateWithFlags(&d->stream, cudaStreamNonBlocking));
    for (auto& e : d->ev) DIT_CUDA(cudaEventCreate(&e));
    DIT_CUDA(cudaMalloc((void**) &d->d_w, d->weight_bytes));
    DIT_CUDA(cudaMalloc((void**) &d->d_norm, d->norm_bytes));
    constexpr size_t CHUNK = 32u << 20;
    uint8_t* pinned[2] = {nullptr, nullptr};
    cudaEvent_t pin_ev[2] = {nullptr, nullptr};
    std::string err;
    for (int i = 0; i < 2 && err.empty(); ++i) {
        if (cudaMallocHost((void**) &pinned[i], CHUNK) != cudaSuccess) err = "qi21_dit: cudaMallocHost failed";
        else if (cudaEventCreateWithFlags(&pin_ev[i], cudaEventDisableTiming) != cudaSuccess) err = "qi21_dit: event";
    }
    int next = 0;
    for (const auto& [t, o] : plan) {
        if (!err.empty()) break;
        err = upload(d, d->d_w + o, d->file->tensor_data(*t), te_row_bytes(t->type, (int64_t) t->shape[0]) * t->shape[1],
                     pinned, pin_ev, CHUNK, next);
    }
    if (err.empty())
        err = upload(d, reinterpret_cast<uint8_t*>(d->d_norm), reinterpret_cast<const uint8_t*>(norms.data()),
                     d->norm_bytes, pinned, pin_ev, CHUNK, next);
    if (err.empty() && cudaStreamSynchronize(d->stream) != cudaSuccess) err = "qi21_dit: upload failed";
    for (int i = 0; i < 2; ++i) {
        if (pinned[i]) cudaFreeHost(pinned[i]);
        if (pin_ev[i]) cudaEventDestroy(pin_ev[i]);
    }
    if (err.empty() && d->f16_resident) err = to_f16_resident(d);
    d->open_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return err;
}

std::string alloc(Qi21Dit* d, int64_t T, int64_t L) {
    if (T <= d->t_cap && L <= d->l_cap) return std::string();
    for (void* p : d->work) cudaFree(p);
    d->work.clear();
    d->work_bytes = 0;
    d->t_cap = std::max(T, d->t_cap);
    d->l_cap = std::max(L, d->l_cap);
    d->prefix_len = -1;                               // the cache lived in the old buffers
    const int64_t R = std::max(d->t_cap, d->l_cap), Lc = std::max<int64_t>(d->l_cap, 1);
    const int64_t C = d->C, F = d->F, H = d->H;
    auto get = [&](auto*& p, size_t bytes) -> std::string {
        bytes = (bytes + 255) / 256 * 256;
        void* q = nullptr;
        DIT_CUDA(cudaMalloc(&q, bytes));
        d->work.push_back(q);
        d->work_bytes += bytes;
        p = static_cast<std::remove_reference_t<decltype(p)>>(q);
        return std::string();
    };
    const size_t f4 = sizeof(float), h2 = sizeof(uint16_t);
    DIT_TRY(get(d->x, R * C * f4));
    DIT_TRY(get(d->h, R * C * f4));
    DIT_TRY(get(d->gu, R * std::max<int64_t>({2 * F, 3 * C, d->ctx_dim}) * f4));
    DIT_TRY(get(d->h16, R * C * h2));
    DIT_TRY(get(d->a16, R * C * h2));
    DIT_TRY(get(d->m16, R * F * h2));
    DIT_TRY(get(d->q16, H * R * HD * f4));
    DIT_TRY(get(d->kw, H * (d->l_cap + d->t_cap) * HD * h2));
    DIT_TRY(get(d->vw, H * (d->l_cap + d->t_cap) * HD * h2));
    DIT_TRY(get(d->pk, (size_t) d->n_layer * H * Lc * HD * h2));
    DIT_TRY(get(d->pv, (size_t) d->n_layer * H * Lc * HD * h2));
    DIT_TRY(get(d->pe_img, d->t_cap * (HD / 2) * 4 * f4));
    DIT_TRY(get(d->pe_txt, Lc * (HD / 2) * 4 * f4));
    DIT_TRY(get(d->latent, d->t_cap * d->in_ch * f4));
    DIT_TRY(get(d->outp, d->t_cap * d->out_ch * f4));
    DIT_TRY(get(d->ctx_in, Lc * d->ctx_dim * f4));
    DIT_TRY(get(d->tin, 2 * f4));
    DIT_TRY(get(d->emb, 2 * d->t_dim * f4));
    DIT_TRY(get(d->te1, 2 * C * f4));
    DIT_TRY(get(d->te2, 2 * C * f4));
    DIT_TRY(get(d->mod, 2 * 4 * C * f4));
    DIT_TRY(get(d->scale, C * f4));
    return std::string();
}

const uint8_t* wp(const Qi21Dit* d, const W2& w) { return d->d_w + w.off; }

// y[n][N] (stride ldy) = x[n][K] (stride ldx) . W^T
std::string mm(Qi21Dit* d, const W2& w, const float* x, int64_t ldx, int64_t n, float* y, int64_t ldy) {
    return dit_gemm(d->stream, w.type, wp(d, w), w.K, w.N, x, ldx, n, y, ldy);
}

// the same with f16 rows (row stride ldx elements) written by the kernel before the GEMM
std::string mm16(Qi21Dit* d, const W2& w, const uint16_t* x16, int64_t ldx, int64_t n, float* y, int64_t ldy) {
    return dit_gemm_x16(d->stream, w.type, wp(d, w), w.K, w.N, x16, ldx, n, y, ldy);
}

// the 32 blocks over rows [0, n) of x; prefix: the text (mod row 1, causal, K / V into the prefix cache); otherwise the
// image tokens (mod row 0) attending [cached text K / V of the block (L rows); their own].
std::string blocks_pass(Qi21Dit* d, int64_t n, bool prefix, const float* pe, int64_t L) {
    void* s = d->stream;
    const int64_t C = d->C, F = d->F, H = d->H, D = HD;
    const float* m = d->mod + (prefix ? 4 * C : 0);
    const float *m0 = m, *m1 = m + C, *m2 = m + 2 * C, *m3 = m + 3 * C;
    const int64_t kvt = prefix ? n : L + n;
    const bool fast = !prefix && d->fast_attn;
    float* X = d->x;
    float* Hh = d->h;
    const size_t hb = sizeof(uint16_t);
    DIT_K(qi21::res_ln_mod_f16(s, X, nullptr, nullptr, d->h16, m0, (int) n, (int) C, EPS));
    for (int i = 0; i < d->n_layer; ++i) {
        const Block& B = d->blocks[i];
        const float *qs = d->gu, *ks = d->gu + C, *vs = d->gu + 2 * C;
        int64_t ld = 3 * C;
        if (B.fused) {
            DIT_TRY(mm16(d, B.qkv, d->h16, C, n, d->gu, 3 * C));
        } else {
            W2 q = B.qkv;
            q.N = C;
            DIT_TRY(mm16(d, q, d->h16, C, n, d->gu, 3 * C));
            DIT_TRY(mm16(d, B.k, d->h16, C, n, d->gu + C, 3 * C));
            DIT_TRY(mm16(d, B.v, d->h16, C, n, d->gu + 2 * C, 3 * C));
        }
        DIT_K(qi21::qkv_post(s, qs, ks, vs, (int) ld, d->d_norm + B.nq, d->d_norm + B.nk, pe, d->q16, fast ? 1 : 0,
                             (int) n, d->kw, d->vw, (int) kvt, (int) (prefix ? 0 : L), (int) n, (int) H, EPS));
        const size_t pbytes = (size_t) L * D * hb;                    // one head's prefix rows
        uint16_t* pkl = d->pk + (size_t) i * H * std::max<int64_t>(d->l_cap, 1) * D;
        uint16_t* pvl = d->pv + (size_t) i * H * std::max<int64_t>(d->l_cap, 1) * D;
        if (prefix) {
            for (int64_t hh = 0; hh < H; ++hh) {
                DIT_CUDA(cudaMemcpyAsync(pkl + hh * n * D, d->kw + hh * kvt * D, n * D * hb, cudaMemcpyDeviceToDevice,
                                         d->stream));
                DIT_CUDA(cudaMemcpyAsync(pvl + hh * n * D, d->vw + hh * kvt * D, n * D * hb, cudaMemcpyDeviceToDevice,
                                         d->stream));
            }
            DIT_K(qi21::attn_exact_f16(s, d->q16, (int) n, d->kw, d->vw, (int) kvt, d->a16, (int) H, (int) n, (int) n, 1));
        } else {
            for (int64_t hh = 0; hh < H && L > 0; ++hh) {
                DIT_CUDA(cudaMemcpyAsync(d->kw + hh * kvt * D, pkl + hh * L * D, pbytes, cudaMemcpyDeviceToDevice,
                                         d->stream));
                DIT_CUDA(cudaMemcpyAsync(d->vw + hh * kvt * D, pvl + hh * L * D, pbytes, cudaMemcpyDeviceToDevice,
                                         d->stream));
            }
            if (fast)
                DIT_K(qi21::attn_fast_f16(s, d->q16, (int) n, d->kw, d->vw, (int) kvt, d->a16, (int) H, (int) n, (int) kvt));
            else
                DIT_K(qi21::attn_exact_f16(s, d->q16, (int) n, d->kw, d->vw, (int) kvt, d->a16, (int) H, (int) n, (int) kvt,
                                       0));
        }
        DIT_TRY(mm16(d, B.out, d->a16, C, n, Hh, C));
        DIT_K(qi21::res_ln_mod_f16(s, X, Hh, m1, d->h16, m2, (int) n, (int) C, EPS));
        DIT_TRY(mm16(d, B.gate_up, d->h16, C, n, d->gu, 2 * F));
        DIT_K(qi21::swiglu_f16(s, d->gu, d->m16, (int) n, (int) F));
        DIT_TRY(mm16(d, B.down, d->m16, F, n, Hh, C));
        const bool last = i + 1 == d->n_layer;
        DIT_K(qi21::res_ln_mod_f16(s, X, Hh, m3, last ? nullptr : d->h16, m0, (int) n, (int) C, EPS));
    }
    return std::string();
}

std::string step_impl(Qi21Dit* d, const float* latent, int W, int Hd, const float* cond, int L, float t, float* vel,
                      Qi21DitStats* st) {
    const int64_t T = (int64_t) W * Hd, C = d->C;
    DIT_CUDA(cudaSetDevice(d->device));
    DIT_TRY(alloc(d, T, L));
    void* s = d->stream;
    DIT_CUDA(cudaEventRecord(d->ev[0], d->stream));
    // rope tables (sd.cpp's layout and embedding, src/image/dit/qi21_rope.cpp: byte-identical to sd.cpp's)
    const std::vector<float> pe = qi21_rope_table(qi21_positions(L, Hd, W));
    const size_t per_tok = (HD / 2) * 4;
    if (pe.size() != (size_t) (L + T) * per_tok) return "qi21_dit: unexpected rope table size";
    DIT_CUDA(cudaMemcpyAsync(d->pe_img, pe.data() + (size_t) L * per_tok, T * per_tok * sizeof(float),
                             cudaMemcpyHostToDevice, d->stream));
    // prelude
    const float tv[2] = {t, 0.f};
    DIT_CUDA(cudaMemcpyAsync(d->tin, tv, sizeof(tv), cudaMemcpyHostToDevice, d->stream));
    DIT_K(qi21::timestep_embedding(s, d->tin, d->emb, 2, d->t_dim, 10000.f));
    DIT_TRY(mm(d, d->lin1, d->emb, d->t_dim, 2, d->te1, C));
    DIT_K(qi21::silu(s, d->te1, 2 * C));
    DIT_TRY(mm(d, d->lin2, d->te1, C, 2, d->te2, C));
    DIT_K(qi21::silu(s, d->te2, 2 * C));
    DIT_TRY(mm(d, d->modw, d->te2, C, 2, d->mod, 4 * C));
    DIT_TRY(mm(d, d->norm_out, d->te2, C, 1, d->scale, C));
    // the text prefix, once per condition
    uint64_t key = 1469598103934665603ull;
    {
        const unsigned char* p = reinterpret_cast<const unsigned char*>(cond);
        const size_t nb = (size_t) L * d->ctx_dim * sizeof(float);
        for (size_t i = 0; i < nb; ++i) key = (key ^ p[i]) * 1099511628211ull;
    }
    const bool run_prefix = key != d->prefix_key || d->prefix_len != L;
    if (run_prefix) {
        DIT_CUDA(cudaMemcpyAsync(d->pe_txt, pe.data(), (size_t) L * per_tok * sizeof(float), cudaMemcpyHostToDevice,
                                 d->stream));
        DIT_CUDA(cudaMemcpyAsync(d->ctx_in, cond, (size_t) L * d->ctx_dim * sizeof(float), cudaMemcpyHostToDevice,
                                 d->stream));
        DIT_K(qi21::rms_norm_w1(s, d->ctx_in, d->d_norm + d->text_norm, d->gu, L, d->ctx_dim, EPS));
        DIT_TRY(mm(d, d->txt_in1, d->gu, d->ctx_dim, L, d->h, C));
        DIT_K(qi21::gelu_tanh(s, d->h, (int64_t) L * C));
        DIT_TRY(mm(d, d->txt_in2, d->h, C, L, d->x, C));
        DIT_TRY(blocks_pass(d, L, true, d->pe_txt, 0));
        d->prefix_key = key;
        d->prefix_len = L;
    }
    DIT_CUDA(cudaEventRecord(d->ev[1], d->stream));
    // the image tokens: patch 1, token h * W + w, channels
    {
        std::vector<float> tok((size_t) T * d->in_ch);
        for (int64_t c = 0; c < d->in_ch; ++c)
            for (int64_t i = 0; i < T; ++i) tok[(size_t) i * d->in_ch + c] = latent[(size_t) c * T + i];
        DIT_CUDA(cudaMemcpyAsync(d->latent, tok.data(), tok.size() * sizeof(float), cudaMemcpyHostToDevice, d->stream));
        DIT_TRY(mm(d, d->img_in, d->latent, d->in_ch, T, d->x, C));
        DIT_CUDA(cudaStreamSynchronize(d->stream));   // tok leaves scope
    }
    DIT_TRY(blocks_pass(d, T, false, d->pe_img, L));
    // head: LN(x) (1 + scale) -> proj_out
    DIT_K(qi21::res_ln_mod_f16(s, d->x, nullptr, nullptr, d->h16, d->scale, (int) T, (int) C, EPS));
    DIT_TRY(mm16(d, d->proj_out, d->h16, C, T, d->outp, d->out_ch));
    DIT_CUDA(cudaEventRecord(d->ev[2], d->stream));
    std::vector<float> o((size_t) T * d->out_ch);
    DIT_CUDA(cudaMemcpyAsync(o.data(), d->outp, o.size() * sizeof(float), cudaMemcpyDeviceToHost, d->stream));
    DIT_CUDA(cudaStreamSynchronize(d->stream));
    for (int64_t c = 0; c < d->out_ch; ++c)
        for (int64_t i = 0; i < T; ++i) vel[(size_t) c * T + i] = o[(size_t) i * d->out_ch + c];
    const size_t total = d->weight_bytes + d->norm_bytes + d->work_bytes + dit_gemm_buffer_bytes();
    d->peak_bytes = std::max(d->peak_bytes, total);
    if (st) {
        float ms = 0, pms = 0;
        cudaEventElapsedTime(&ms, d->ev[0], d->ev[2]);
        cudaEventElapsedTime(&pms, d->ev[0], d->ev[1]);
        st->step_ms = ms;
        st->prefix_ran = run_prefix;
        st->prefix_ms = run_prefix ? pms : 0;
        st->weight_bytes = d->weight_bytes + d->norm_bytes;
        st->work_bytes = d->work_bytes + dit_gemm_buffer_bytes();
        st->peak_bytes = d->peak_bytes;
    }
    return std::string();
}

}  // namespace

Qi21Dit* qi21_dit_open(const std::string& gguf_path, int device, std::string* err, int flags) {
    auto* d = new Qi21Dit();
    d->f16_resident = (flags & QI21_DIT_F16_RESIDENT) != 0;
    std::string e;
    try {
        e = open_impl(d, gguf_path, device);
    } catch (const std::exception& x) {
        e = std::string("qi21_dit: ") + x.what();
    }
    if (!e.empty()) {
        if (err) *err = e;
        qi21_dit_close(d);
        return nullptr;
    }
    return d;
}

int qi21_dit_in_channels(const Qi21Dit* d) { return d->in_ch; }
int qi21_dit_out_channels(const Qi21Dit* d) { return d->out_ch; }
int qi21_dit_context_dim(const Qi21Dit* d) { return d->ctx_dim; }

std::string qi21_dit_step(Qi21Dit* d, const float* latent, int W, int H, const float* cond, int L, float t, float* velocity,
                          Qi21DitStats* stats) {
    if (!d || W <= 0 || H <= 0 || L <= 0) return "qi21_dit: bad arguments";
    return step_impl(d, latent, W, H, cond, L, t, velocity, stats);
}

std::string qi21_dit_open_report(const Qi21Dit* d) {
    char b[512];
    int fused = 0;
    for (const auto& B : d->blocks) fused += B.fused;
    std::snprintf(b, sizeof b,
                  "%d blocks, width %d, ffn %d, heads %d, in %d, out %d, context %d; weights %.3f GB + norms %.1f KB on "
                  "device %d; q|k|v fused in %d blocks; attention %s; open %.0f ms; %zu tensors not used",
                  d->n_layer, d->C, d->F, d->H, d->in_ch, d->out_ch, d->ctx_dim, d->weight_bytes / 1e9,
                  d->norm_bytes / 1e3, d->device, fused, d->fast_attn ? "fast (mma / WMMA)" : "exact (f32)", d->open_ms,
                  d->unused.size());
    std::string r = b;
    for (size_t i = 0; i < d->unused.size() && i < 8; ++i) r += (i ? ", " : ": ") + d->unused[i];
    if (d->f16_resident) {
        std::snprintf(b, sizeof b, "; weights f16-resident (decoded once at open, %.0f ms)", d->decode_ms);
        r += b;
    }
    return r;
}

void qi21_dit_close(Qi21Dit* d) {
    if (!d) return;
    for (void* p : d->work) cudaFree(p);
    if (d->d_w) cudaFree(d->d_w);
    if (d->d_norm) cudaFree(d->d_norm);
    for (auto& e : d->ev)
        if (e) cudaEventDestroy(e);
    if (d->stream) cudaStreamDestroy(d->stream);
    dit_gemm_release();
    delete d;
}

}  // namespace strata::image
