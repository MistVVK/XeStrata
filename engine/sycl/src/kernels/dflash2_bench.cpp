// sycl/src/kernels/dflash2_bench.cpp - DF2-SYCL: what one DFlash2 drafter round costs on an Intel Arc (the Q4_0 pack's real shapes), and checks of the SYCL kernels.
//   dflash2_bench [test|bench]     (default bench)
//   env DF2B_POS (ring position, default 600), DF2B_ITERS (default 100), DF2B_NNEW (context rows, default 3), DF2B_ND (drafts, 1..7, default 7),
//       DF2B_STANDIN_MB (a memory-bound kernel run before each timed round, default 0), DF2B_GAPMS (host gap between rounds, default 0),
//       DF2B_PROFILE=1 (a marker after every launch: ms per kernel; direct launches only), DF2B_GRAPH=0 (skip the sycl graph replay)
// One round = the context update for n_new rows (fc 12800 -> 2560, per layer K/V projections, norm + rope, ring writes) + the 8-row block (5 layers: dynamic
// conv projections, q/k/v, norm + rope, windowed attention over the ring, o, conv + rms, gate/up, swiglu, down) + the head over the draft vocabulary
// (nd rows, Q8_0 rows through the engine's native_mmvq) + top-16 + the candidate selector.  The weights are random bytes in the pack's real sizes
// (tools/dflash2_pack.py --q4_0: 268 MB of Q4_0 matrices + 254 MB of bf16 selector codebooks); the embedding is a small table gather.
#include "strata/kernels/dflash2.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/sycl_queue.hpp"

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace strata::kernels;
namespace sx = sycl::ext::oneapi::experimental;

namespace {
constexpr int N = 2560, L = 5, I = 7680, NH = 32, NKV = 8, HD = 128, B = 8, RING = 2064, WINDOW = 2048, NDV = 106299, RANK = 256, VOCAB = 248320, GROUP = 16;
constexpr float EPS = 1e-6f;
constexpr int GGML_Q8_0 = 8;

uint32_t rs = 12345;
uint32_t rnd() { rs = rs * 1664525u + 1013904223u; return rs >> 8; }
float frand() { return (float) (rnd() & 0xffff) / 65536.0f - 0.5f; }
uint16_t f2h(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000, e = ((x >> 23) & 0xff), m = x & 0x7fffff;
    if (e < 103) return (uint16_t) s;
    if (e > 142) return (uint16_t) (s | 0x7c00);
    if (e < 113) { const uint32_t mm = (m | 0x800000) >> (126 - e); return (uint16_t) (s | (mm >> 13)); }
    return (uint16_t) (s | ((e - 112) << 10) | (m >> 13));
}
float h2f(uint16_t h) {
    const uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 31, m = h & 1023;
    uint32_t x;
    if (e == 0) { if (m == 0) x = s; else { float f = m / 1024.0f / 16384.0f; std::memcpy(&x, &f, 4); x |= s; } }
    else x = s | ((e + 112) << 23) | (m << 13);
    float f; std::memcpy(&f, &x, 4); return f;
}
double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
double pct(std::vector<double> v, double p) { std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, (size_t) (p * v.size()))]; }
int envi(const char* k, int d) { const char* e = std::getenv(k); return e && e[0] ? std::atoi(e) : d; }
double envd(const char* k, double d) { const char* e = std::getenv(k); return e && e[0] ? std::atof(e) : d; }
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

sycl::queue* Q = nullptr;
template <class T> T* dev(size_t n) { T* p = sycl::malloc_device<T>(n, *Q); if (!p) { std::fprintf(stderr, "out of device memory (%zu bytes)\n", n * sizeof(T)); std::exit(2); } return p; }
template <class T> void up(T* d, const std::vector<T>& h) { Q->memcpy(d, h.data(), h.size() * sizeof(T)).wait(); }
template <class T> std::vector<T> down(const T* d, size_t n) { std::vector<T> h(n); Q->memcpy(h.data(), d, n * sizeof(T)).wait(); return h; }

// A Q4_0 matrix in the tiled layout: n_out * n_in / 2 random bytes (any byte is two valid nibbles) then the fp16 scale plane (n_out * n_in / 32 random small halves).
struct Mat4 { uint8_t* w; int n_out, n_in; };
Mat4 make_mat(int n_out, int n_in) {
    const size_t nib = (size_t) n_out * n_in / 2, nsc = (size_t) n_out * n_in / 32;
    std::vector<uint8_t> h(nib + nsc * 2);
    for (size_t i = 0; i < nib; ++i) h[i] = (uint8_t) rnd();
    for (size_t i = 0; i < nsc; ++i) { const uint16_t d = (uint16_t) (0x1800 + (rnd() % 0xC00)); std::memcpy(&h[nib + i * 2], &d, 2); }
    uint8_t* d = dev<uint8_t>(h.size());
    Q->memcpy(d, h.data(), h.size()).wait();
    return Mat4{d, n_out, n_in};
}
float* make_f32(size_t n, float scale, float off = 0.f) {
    std::vector<float> h(n);
    for (auto& v : h) v = off + scale * frand();
    float* d = dev<float>(n);
    Q->memcpy(d, h.data(), n * 4).wait();
    return d;
}

struct Layer { Mat4 q, k, v, o, gate, up, down, akp, mkp; float *base_a, *base_m, *ln_in, *ln_post, *qn, *kn; };

struct Drafter {
    std::vector<Layer> ly;
    Mat4 fc, hsel;
    float *hidden_norm, *final_norm;
    uint16_t *pred_cb, *succ_cb;
    uint8_t* head_w; int32_t* dvocab;
    float* emb_tab;
    // buffers
    int32_t* d_meta; int32_t* h_meta; int32_t* h_out; float* h_prob; int32_t* d_out;
    float *kring[L], *vring[L]; int64_t* tag; float* rt_ctx; float* rt_blk;
    uint8_t *xq_head, *xqp, *xqc; float *xdp, *xdc;
    float *taps, *fcy, *kc, *vc, *h, *xn, *dyn, *qv, *qn, *kv, *vv, *kblk, *ao, *o, *gate, *up, *mlp, *part, *fn, *logits, *topval, *hsel_out;
    int32_t *toptok; uint32_t* topscratch;
    int nd = 7;
    std::vector<std::pair<std::string, sycl::event>> marks;
    bool profile = false;
    void stamp(const char* name) { if (profile) marks.emplace_back(name, Q->single_task([] {})); }
};

uint8_t* make_head() {   // Q8_0 rows: 34 bytes a block (fp16 d, 32 int8), NDV rows of N
    const size_t blocks = (size_t) NDV * (N / 32);
    std::vector<uint8_t> h(blocks * 34);
    for (size_t b = 0; b < blocks; ++b) {
        const uint16_t d = f2h(0.002f + 0.0004f * (float) (rnd() % 10));
        std::memcpy(&h[b * 34], &d, 2);
        for (int i = 0; i < 32; ++i) h[b * 34 + 2 + i] = (uint8_t) rnd();
    }
    uint8_t* d = dev<uint8_t>(h.size());
    Q->memcpy(d, h.data(), h.size()).wait();
    return d;
}

void build(Drafter& D) {
    D.ly.resize(L);
    for (int l = 0; l < L; ++l) {
        Layer& y = D.ly[l];
        y.q = make_mat(NH * HD, N); y.k = make_mat(NKV * HD, N); y.v = make_mat(NKV * HD, N); y.o = make_mat(N, NH * HD);
        y.gate = make_mat(I, N); y.up = make_mat(I, N); y.down = make_mat(N, I); y.akp = make_mat(640, N); y.mkp = make_mat(640, N);
        y.base_a = make_f32((size_t) 4 * N, 0.2f, 0.5f); y.base_m = make_f32((size_t) 4 * N, 0.2f, 0.5f);
        y.ln_in = make_f32(N, 0.2f, 1.f); y.ln_post = make_f32(N, 0.2f, 1.f); y.qn = make_f32(HD, 0.2f, 1.f); y.kn = make_f32(HD, 0.2f, 1.f);
    }
    D.fc = make_mat(N, 5 * N);
    D.hsel = make_mat(RANK, N);
    D.hidden_norm = make_f32(N, 0.2f, 1.f); D.final_norm = make_f32(N, 0.2f, 1.f);
    {
        std::vector<uint16_t> cb((size_t) VOCAB * RANK);
        for (auto& v : cb) v = (uint16_t) (0x3c00 + (rnd() & 0x1ff) - 0x100 - 0x300);   // bf16 near 2^-4 .. 2^-3, either sign mixed below
        for (size_t i = 0; i < cb.size(); i += 5) cb[i] |= 0x8000;
        D.pred_cb = dev<uint16_t>(cb.size()); up(D.pred_cb, cb);
        for (size_t i = 0; i < cb.size(); i += 3) cb[i] ^= 0x8000;
        D.succ_cb = dev<uint16_t>(cb.size()); up(D.succ_cb, cb);
    }
    D.head_w = make_head();
    {
        std::vector<int32_t> v(NDV);
        for (int i = 0; i < NDV; ++i) v[i] = (int32_t) (((int64_t) i * 2654435761ll) % VOCAB);   // draft vocabulary -> token id
        D.dvocab = dev<int32_t>(NDV); up(D.dvocab, v);
    }
    D.emb_tab = make_f32((size_t) 64 * N, 1.0f);
    D.d_meta = dev<int32_t>(32 + 2048);   // 32 ints, then the rope tables
    D.h_meta = (int32_t*) strata::host_malloc_polled((32 + 2048) * 4, *Q);   // 32 ints, then the rope tables of the 16 rows (host-computed: no device FP64)
    D.h_out = (int32_t*) strata::host_malloc_polled(64 * 4, *Q);
    std::memset(D.h_meta, 0, (32 + 2048) * 4); std::memset(D.h_out, 0, 256);
    D.h_prob = (float*) D.h_out + 16;
    D.d_out = dev<int32_t>(64); Q->memset(D.d_out, 0, 256).wait();
    for (int l = 0; l < L; ++l) { D.kring[l] = make_f32((size_t) RING * NKV * HD, 2.f); D.vring[l] = make_f32((size_t) RING * NKV * HD, 2.f); }
    D.tag = dev<int64_t>(RING);
    D.rt_ctx = (float*) (D.d_meta + 32);
    D.rt_blk = D.rt_ctx + B * 128;
    D.xq_head = dev<uint8_t>(native_q8_1_bytes(N, B) + 4096);
    D.xqp = dev<uint8_t>((size_t) 5 * N * 8 + 4096); D.xdp = dev<float>((size_t) 5 * N / 32 * 8 + 64);
    D.xqc = dev<uint8_t>((size_t) 5 * N * 8 + 4096); D.xdc = dev<float>((size_t) 5 * N / 32 * 8 + 64);
    Q->memset(D.xqp, 0, (size_t) 5 * N * 8 + 4096); Q->memset(D.xdp, 0, ((size_t) 5 * N / 32 * 8 + 64) * 4);
    Q->memset(D.xqc, 0, (size_t) 5 * N * 8 + 4096); Q->memset(D.xdc, 0, ((size_t) 5 * N / 32 * 8 + 64) * 4);
    D.taps = make_f32((size_t) B * 5 * N, 1.0f);
    D.fcy = dev<float>((size_t) B * N); D.kc = dev<float>((size_t) B * NKV * HD); D.vc = dev<float>((size_t) B * NKV * HD);
    D.h = dev<float>((size_t) B * N); D.xn = dev<float>((size_t) B * N); D.dyn = dev<float>((size_t) B * 640);
    D.qv = dev<float>((size_t) B * NH * HD); D.qn = dev<float>((size_t) B * NH * HD); D.kv = dev<float>((size_t) B * NKV * HD); D.vv = dev<float>((size_t) B * NKV * HD);
    D.kblk = dev<float>((size_t) B * NKV * HD); D.ao = dev<float>((size_t) B * NH * HD); D.o = dev<float>((size_t) B * N);
    D.gate = dev<float>((size_t) B * I); D.up = dev<float>((size_t) B * I); D.mlp = dev<float>((size_t) B * N);
    D.part = dev<float>((size_t) kDf2PartChunks * 8 * 32 * 130); D.fn = dev<float>((size_t) B * N);
    D.logits = dev<float>((size_t) 7 * NDV); D.topval = dev<float>(7 * 16); D.toptok = dev<int32_t>(7 * 16); D.hsel_out = dev<float>(7 * RANK);
    D.topscratch = dev<uint32_t>((size_t) 7 * kDf2TopBlocks * kDf2TopK * 2);
    Q->memset(D.tag, 0xff, RING * 8).wait();
    Q->wait();
}

void reset_ring(Drafter& D, int pos) {
    std::vector<int64_t> t(RING, -1);
    for (int p = std::max(0, pos - (RING - 16)); p < pos; ++p) t[p % RING] = ((int64_t) 1 << 32) | p;
    Q->memcpy(D.tag, t.data(), RING * 8).wait();
}

void set_meta(Drafter& D, int n_new, int64_t pos0, int anchor) {
    const int64_t s = pos0 + n_new;
    for (int i = 0; i < 8; ++i) { D.h_meta[i] = (int32_t) (pos0 + std::min(i, n_new - 1)); D.h_meta[8 + i] = (int32_t) (s + i); }
    D.h_meta[16] = anchor; D.h_meta[17] = (int32_t) s; D.h_meta[18] = 1; D.h_meta[24] = anchor;
    for (int i = 1; i < 8; ++i) D.h_meta[24 + i] = 7;
    df2_rope_table_cpu(D.h_meta, 16, 10000.0f, (float*) (D.h_meta + 32));
    for (int i = 0; i < 16; ++i) ((volatile int32_t*) D.h_out)[i] = -1;
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

void gemm1(Drafter&, const Mat4& m, float* y, int nc, const uint8_t* xq, const float* xd) {
    Df2Mat mats[1] = {Df2Mat{m.w, y, m.n_out, true}};
    df2_gemm(mats, 1, xq, xd, m.n_in, nc, Q);
}
void gemm2(const Mat4& a, const Mat4& b, float* ya, float* yb, int nc, const uint8_t* xq, const float* xd) {
    Df2Mat mats[2] = {Df2Mat{a.w, ya, a.n_out, true}, Df2Mat{b.w, yb, b.n_out, true}};
    df2_gemm(mats, 2, xq, xd, a.n_in, nc, Q);
}
void gemm3(const Mat4& a, const Mat4& b, const Mat4& c, float* ya, float* yb, float* yc, int nc, const uint8_t* xq, const float* xd) {
    Df2Mat mats[3] = {Df2Mat{a.w, ya, a.n_out, true}, Df2Mat{b.w, yb, b.n_out, true}, Df2Mat{c.w, yc, c.n_out, true}};
    df2_gemm(mats, 3, xq, xd, a.n_in, nc, Q);
}

bool g_io = true;
int g_meta_kernel = 0;   // DF2B_METAKERNEL=1: a kernel copies the meta from host memory
int g_host_out = 0;   // DF2B_HOSTOUT=1: the select kernel writes the host buffer itself
std::string g_dup;   // DF2B_DUP=name: every launch stamped `name` runs twice (the marginal cost of that kernel in the round)
int g_dupn = 1;   // DF2B_DUPN: how many extra launches of the DF2B_DUP kernel (layer 0 only when > 1)
int g_layer = 0;
#define DO(name, call) do { call; if (!g_dup.empty() && g_dup == name && (g_dupn == 1 || g_layer == 0)) { for (int rep = 0; rep < g_dupn; ++rep) { call; } } D.stamp(name); } while (0)
void record_context(Drafter& D, int n) {
    const int tap_n = 5 * N;
    DO("quant_rows", df2_quant_rows(D.taps, tap_n, n, D.xqc, D.xdc, Q));
    DO("gemm:fc", gemm1(D, D.fc, D.fcy, n, D.xqc, D.xdc));
    DO("rms_quant", df2_rms_quant(D.fcy, D.hidden_norm, nullptr, n, N, EPS, D.xqc, D.xdc, Q));
    for (int l = 0; l < L; ++l) {
        DO("gemm:kp", gemm2(D.ly[l].k, D.ly[l].v, D.kc, D.vc, n, D.xqc, D.xdc));
        DO("kv_ring", df2_kv_ring(D.kc, D.vc, D.ly[l].kn, D.d_meta + 0, D.rt_ctx, D.kring[l], D.vring[l], RING, n, NKV, EPS, Q, D.tag, D.d_meta + 18));
    }
}

// the stand-in for the engine's embedding gather: 8 rows of a small table
void embed_stub(Drafter& D) {
    const float* tab = D.emb_tab; const int32_t* tok = D.d_meta + 24; float* out = D.h;
    Q->parallel_for(sycl::nd_range<1>((size_t) B * N, 256), [=](sycl::nd_item<1> it) {
        const int i = (int) it.get_global_id(0), r = i / N, c = i % N;
        out[i] = tab[(size_t) (tok[r] & 63) * N + c];
    });
}

void record_block(Drafter& D) {
    DO("embed", embed_stub(D));
    DO("rms_quant", df2_rms_quant(D.h, D.ly[0].ln_in, D.xn, B, N, EPS, D.xqp, D.xdp, Q));
    for (int l = 0; l < L; ++l) {
        Layer& y = D.ly[l]; g_layer = l;
        DO("gemm:akp", gemm1(D, y.akp, D.dyn, B, D.xqp, D.xdp));
        DO("conv_quant", df2_conv_quant(D.xn, D.dyn, 640, y.base_a, B, N, GROUP, D.xqp, D.xdp, Q));
        DO("gemm:qkv", gemm3(y.q, y.k, y.v, D.qv, D.kv, D.vv, B, D.xqp, D.xdp));
        DO("norm_rope_qk", df2_norm_rope_qk(D.qv, D.kv, y.qn, y.kn, D.rt_blk, D.qn, D.kblk, B, NH, NKV, EPS, Q));
        DO("attention", df2_attention(D.qn, D.kring[l], D.vring[l], D.kblk, D.vv, D.d_meta + 17, RING, WINDOW, D.part, D.xqp, D.xdp, Q, D.tag, D.d_meta + 18));
        DO("gemm:o", gemm1(D, y.o, D.o, B, D.xqp, D.xdp));
        DO("conv_rms_quant", df2_conv_rms_quant(D.o, D.dyn + 320, 640, y.base_a + 2 * N, D.h, y.ln_post, D.xn, B, N, GROUP, EPS, D.xqp, D.xdp, Q));
        DO("gemm:mkp", gemm1(D, y.mkp, D.dyn, B, D.xqp, D.xdp));
        DO("conv_quant", df2_conv_quant(D.xn, D.dyn, 640, y.base_m, B, N, GROUP, D.xqp, D.xdp, Q));
        DO("gemm:gate_up", gemm2(y.gate, y.up, D.gate, D.up, B, D.xqp, D.xdp));
        DO("swiglu_quant", df2_swiglu_quant(D.gate, D.up, B, I, D.xqp, D.xdp, Q));
        DO("gemm:down", gemm1(D, y.down, D.mlp, B, D.xqp, D.xdp));
        const float* next_gamma = l + 1 < L ? D.ly[l + 1].ln_in : D.final_norm;
        DO("conv_rms_quant", df2_conv_rms_quant(D.mlp, D.dyn + 320, 640, y.base_m + 2 * N, D.h, next_gamma, l + 1 < L ? D.xn : D.fn, B, N, GROUP, EPS, D.xqp, D.xdp, Q));
    }
    DO("quant_q8_1", native_quantize_q8_1(D.fn + N, D.xq_head, N, D.nd, Q));
    DO("quant_rows", df2_quant_rows(D.fn + N, N, D.nd, D.xqp, D.xdp, Q));
    DO("head", native_mmvq(GGML_Q8_0, D.head_w, D.xq_head, D.logits, N, NDV, D.nd, Q));
    DO("topk", df2_topk(D.logits, NDV, D.dvocab, D.nd, D.topscratch, D.topval, D.toptok, Q));
    DO("gemm:hsel", gemm1(D, D.hsel, D.hsel_out, D.nd, D.xqp, D.xdp));
    DO("select", df2_select(D.hsel_out, D.topval, D.toptok, D.pred_cb, D.succ_cb, RANK, D.nd, D.d_meta + 16, g_host_out ? D.h_out : D.d_out, g_host_out ? (float*) D.h_out + 16 : (float*) D.d_out + 16, Q));
    if (g_host_out == 0 && g_io) DO("d2h", Q->memcpy(D.h_out, D.d_out, 64 * 4));
}

void record_round(Drafter& D, int n_new, bool io = true) {
    g_io = io;
    // meta + rope tables to the device with one copy (no kernel touches host memory: a kernel that reads or writes host USM slows every later kernel of the process on the A750)
    if (!io) {
    } else if (g_meta_kernel) {
        const int32_t* hm = D.h_meta; int32_t* dm = D.d_meta;
        Q->parallel_for(sycl::range<1>(32 + 2048), [=](sycl::id<1> i) { dm[i] = hm[i]; });
    } else {
        Q->memcpy(D.d_meta, D.h_meta, (32 + 2048) * 4);
    }
    D.stamp("meta_rope");
    record_context(D, n_new);
    record_block(D);
}

uint64_t checksum(const void* p, size_t n) { uint64_t h = 1469598103934665603ull; for (size_t i = 0; i < n; ++i) h = (h ^ ((const uint8_t*) p)[i]) * 1099511628211ull; return h; }

// ---------------------------------------------------------------- tests
int g_fail = 0;
void check(bool ok, const char* what) { std::printf("  %-64s %s\n", what, ok ? "ok" : "FAIL"); if (!ok) ++g_fail; }

// CPU reference of the Q4_0 GEMM, bitwise (the lane partial sums over the steps in order, fma, the xor shuffle tree)
void gemm_ref(const std::vector<uint8_t>& w, int n_out, int n_in, const std::vector<int8_t>& xq, const std::vector<float>& xd, int nc, std::vector<float>& y) {
    const int nb = n_in / 32, steps = nb / 16;
    const size_t nib = (size_t) n_out * n_in / 2;
    y.assign((size_t) nc * n_out, 0.f);
    for (int row = 0; row < n_out; ++row) {
        const int g = row / 8, rr = row % 8;
        for (int c = 0; c < nc; ++c) {
            float lane[32];
            for (int l = 0; l < 32; ++l) {
                float acc = 0.f;
                for (int st = 0; st < steps; ++st) {
                    const uint8_t* wp = &w[(size_t) g * 8 * (n_in / 2) + (size_t) st * 2048 + (size_t) rr * 256 + (size_t) l * 8];
                    uint32_t rx, ry; std::memcpy(&rx, wp, 4); std::memcpy(&ry, wp + 4, 4);
                    const uint32_t ws[4] = {rx & 0x0F0F0F0Fu, (rx >> 4) & 0x0F0F0F0Fu, ry & 0x0F0F0F0Fu, (ry >> 4) & 0x0F0F0F0Fu};
                    const int8_t* xp = &xq[(size_t) c * n_in + (size_t) st * 512 + (size_t) l * 16];
                    int isum = 0, xsum = 0;
                    for (int k = 0; k < 4; ++k)
                        for (int b = 0; b < 4; ++b) { isum += (int) ((ws[k] >> (8 * b)) & 0xff) * (int) xp[k * 4 + b]; xsum += (int) xp[k * 4 + b]; }
                    uint16_t dh; std::memcpy(&dh, &w[nib + ((size_t) g * 8 * nb + (size_t) st * 128 + rr * 16 + (l >> 1)) * 2], 2);
                    const float dw = h2f(dh), xs = xd[(size_t) c * nb + st * 16 + (l >> 1)];
                    const float prod = dw * xs;
                    acc = std::fmaf(prod, (float) (isum - 8 * xsum), acc);
                }
                lane[l] = acc;
            }
            for (int o = 16; o > 0; o >>= 1) { float nl[32]; for (int l = 0; l < 32; ++l) nl[l] = lane[l] + lane[l ^ o]; std::memcpy(lane, nl, sizeof lane); }
            y[(size_t) c * n_out + row] = lane[0];
        }
    }
}


// DF-04: the head, top-k and selector for the first nd rows give exactly the first nd rows of the 7-row path (distinct random rows, so the picks differ per row)
void prefix_test() {
    std::printf("DF-04 prefix: head / top-k / selector for nd rows == the first nd rows of the 7-row path (bitwise)\n");
    uint8_t* head_w = make_head();
    std::vector<int32_t> vv(NDV); for (int i = 0; i < NDV; ++i) vv[i] = (int32_t) (((int64_t) i * 2654435761ll) % VOCAB);
    int32_t* dvocab = dev<int32_t>(NDV); up(dvocab, vv);
    std::vector<uint16_t> cb((size_t) VOCAB * RANK);
    for (auto& v : cb) v = (uint16_t) (0x3c00 + (rnd() & 0x1ff) - 0x400); for (size_t i = 0; i < cb.size(); i += 5) cb[i] |= 0x8000;
    uint16_t* pred = dev<uint16_t>(cb.size()); up(pred, cb);
    for (size_t i = 0; i < cb.size(); i += 3) cb[i] ^= 0x8000;
    uint16_t* succ = dev<uint16_t>(cb.size()); up(succ, cb);
    Mat4 hsel = make_mat(RANK, N);
    float* fn = make_f32((size_t) B * N, 1.0f);
    uint8_t* xq_head = dev<uint8_t>(native_q8_1_bytes(N, B) + 4096);
    uint8_t* xqp = dev<uint8_t>((size_t) 5 * N * 8 + 4096); float* xdp = dev<float>((size_t) 5 * N / 32 * 8 + 64);
    float* logits = dev<float>((size_t) 7 * NDV); float* hs = dev<float>(7 * RANK); float* tv = dev<float>(7 * 16); int32_t* tt = dev<int32_t>(7 * 16);
    uint32_t* sc = dev<uint32_t>((size_t) 7 * kDf2TopBlocks * kDf2TopK * 2);
    int32_t* anchor = dev<int32_t>(1); int32_t a5 = 777; Q->memcpy(anchor, &a5, 4).wait();
    int32_t* otok = dev<int32_t>(8); float* oprob = dev<float>(8);
    struct Out { std::vector<float> lg, hs, tv; std::vector<int32_t> tt, ot; std::vector<float> op; };
    auto run = [&](int nd) {
        native_quantize_q8_1(fn + N, xq_head, N, nd, Q);
        df2_quant_rows(fn + N, N, nd, xqp, xdp, Q);
        native_mmvq(GGML_Q8_0, head_w, xq_head, logits, N, NDV, nd, Q);
        df2_topk(logits, NDV, dvocab, nd, sc, tv, tt, Q);
        { Df2Mat m1[1] = {Df2Mat{hsel.w, hs, hsel.n_out, true}}; df2_gemm(m1, 1, xqp, xdp, hsel.n_in, nd, Q); }
        df2_select(hs, tv, tt, pred, succ, RANK, nd, anchor, otok, oprob, Q);
        Q->wait();
        Out o; o.lg = down(logits, (size_t) nd * NDV); o.hs = down(hs, (size_t) nd * RANK); o.tv = down(tv, (size_t) nd * 16); o.tt = down(tt, (size_t) nd * 16);
        o.ot = down(otok, nd); o.op = down(oprob, nd);
        return o;
    };
    const Out full = run(7);
    bool distinct = false; for (int i = 1; i < 7; ++i) distinct |= full.ot[i] != full.ot[0];
    check(distinct, "the 7 picks of the reference differ (the test can tell rows apart)");
    for (int nd : {5, 3, 1}) {
        const Out o = run(nd);
        char buf[96];
        bool ok = std::memcmp(o.lg.data(), full.lg.data(), o.lg.size() * 4) == 0 && std::memcmp(o.hs.data(), full.hs.data(), o.hs.size() * 4) == 0 &&
                  std::memcmp(o.tv.data(), full.tv.data(), o.tv.size() * 4) == 0 && std::memcmp(o.tt.data(), full.tt.data(), o.tt.size() * 4) == 0 &&
                  std::memcmp(o.ot.data(), full.ot.data(), o.ot.size() * 4) == 0 && std::memcmp(o.op.data(), full.op.data(), o.op.size() * 4) == 0;
        std::snprintf(buf, sizeof buf, "nd = %d: logits, selector projection, top-16, picks, probabilities == first %d rows of nd = 7", nd, nd);
        check(ok, buf);
    }
}

int run_tests() {
    std::printf("device: %s\n", Q->get_device().get_info<sycl::info::device::name>().c_str());
    struct Shape { const char* name; int n_out, n_in, nmat; };
    const Shape shapes[] = {{"fc", N, 5 * N, 1}, {"qkv(3 matrices)", NH * HD, N, 3}, {"o", N, NH * HD, 1}, {"gate+up(2)", I, N, 2}, {"down", N, I, 1}, {"akp", 640, N, 1}, {"hsel", RANK, N, 1}};
    const int variants[] = {17, 28, 12, 11, 10, 24, 25, 34, 40, 41, 42, 43};
    std::printf("Q4_0 GEMM against the CPU reference (bitwise), every variant, columns 1..8; and the first n columns of the 8-column launch against the n-column launch (DF-03)\n");
    for (const Shape& s : shapes) {
        const int n_in = s.n_in;
        std::vector<Mat4> mats;
        std::vector<std::vector<uint8_t>> host;
        std::vector<float*> ys;
        for (int m = 0; m < s.nmat; ++m) { mats.push_back(make_mat(s.n_out, n_in)); host.push_back(down(mats.back().w, (size_t) s.n_out * n_in / 2 + (size_t) s.n_out * n_in / 16)); ys.push_back(dev<float>((size_t) 8 * s.n_out)); }
        std::vector<int8_t> xq((size_t) 8 * n_in);
        std::vector<float> xd((size_t) 8 * n_in / 32);
        for (auto& v : xq) v = (int8_t) (rnd() % 255 - 127);
        for (auto& v : xd) v = 0.001f + 0.00002f * (float) (rnd() % 100);
        int8_t* dxq = dev<int8_t>(xq.size()); float* dxd = dev<float>(xd.size());
        Q->memcpy(dxq, xq.data(), xq.size()).wait(); Q->memcpy(dxd, xd.data(), xd.size() * 4).wait();
        bool all_ref = true, all_nc = true;
        std::vector<std::vector<float>> ref(s.nmat);
        for (int m = 0; m < s.nmat; ++m) gemm_ref(host[m], s.n_out, n_in, xq, xd, 8, ref[m]);
        for (int v : variants) {
            df2_gemm_variant(v);
            std::vector<std::vector<float>> full(s.nmat);
            for (int nc = 8; nc >= 1; --nc) {
                std::vector<Df2Mat> dm;
                for (int m = 0; m < s.nmat; ++m) { Q->memset(ys[m], 0, (size_t) 8 * s.n_out * 4); dm.push_back(Df2Mat{mats[m].w, ys[m], s.n_out, true}); }
                df2_gemm(dm.data(), s.nmat, dxq, dxd, n_in, nc, Q);
                Q->wait();
                for (int m = 0; m < s.nmat; ++m) {
                    auto y = down(ys[m], (size_t) nc * s.n_out);
                    if (nc == 8) { full[m] = y; all_ref &= std::memcmp(y.data(), ref[m].data(), y.size() * 4) == 0; }
                    else all_nc &= std::memcmp(y.data(), full[m].data(), y.size() * 4) == 0;
                }
            }
        }
        df2_gemm_variant(0);
        char buf[160];
        std::snprintf(buf, sizeof buf, "%s (%d x %d): 8 variants x 8 NC vs CPU", s.name, s.n_out, n_in); check(all_ref, buf);
        std::snprintf(buf, sizeof buf, "%s: NC < 8 == first NC columns of NC = 8", s.name); check(all_nc, buf);
    }
    // top-16 against the CPU (value descending, ties to the lower index)
    {
        const int rows = 7;
        std::vector<float> lg((size_t) rows * NDV);
        for (auto& v : lg) v = (float) (rnd() % 20000) / 100.f;   // many ties
        float* dl = dev<float>(lg.size()); up(dl, lg);
        int32_t* dmap = dev<int32_t>(NDV); std::vector<int32_t> mp(NDV); for (int i = 0; i < NDV; ++i) mp[i] = i * 2; up(dmap, mp);
        uint32_t* sc = dev<uint32_t>((size_t) rows * kDf2TopBlocks * kDf2TopK * 2); float* tv = dev<float>(rows * 16); int32_t* tt = dev<int32_t>(rows * 16);
        df2_topk(dl, NDV, dmap, rows, sc, tv, tt, Q); Q->wait();
        auto hv = down(tv, rows * 16); auto ht = down(tt, rows * 16);
        bool ok = true;
        for (int r = 0; r < rows; ++r) {
            std::vector<int> idx(NDV); for (int i = 0; i < NDV; ++i) idx[i] = i;
            std::partial_sort(idx.begin(), idx.begin() + 16, idx.end(), [&](int a, int b) { const float va = lg[(size_t) r * NDV + a], vb = lg[(size_t) r * NDV + b]; return va > vb || (va == vb && a < b); });
            for (int k = 0; k < 16; ++k) ok &= hv[r * 16 + k] == lg[(size_t) r * NDV + idx[k]] && ht[r * 16 + k] == idx[k] * 2;
        }
        check(ok, "top-16 of 7 x 106299 logits (ties) == CPU, with the id map");
    }
    // quantize rows against the CPU (exact)
    {
        const int K = N, rows = 5;
        std::vector<float> x((size_t) rows * K); for (auto& v : x) v = 3.f * frand();
        float* dx = dev<float>(x.size()); up(dx, x);
        int8_t* dq = dev<int8_t>((size_t) rows * K); float* dd = dev<float>((size_t) rows * K / 32);
        df2_quant_rows(dx, K, rows, dq, dd, Q); Q->wait();
        auto hq = down(dq, (size_t) rows * K); auto hd = down(dd, (size_t) rows * K / 32);
        bool ok = true;
        for (int r = 0; r < rows; ++r)
            for (int b = 0; b < K / 32; ++b) {
                float amax = 0; for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[(size_t) r * K + b * 32 + i]));
                const float d = amax / 127.f, inv = amax > 0 ? 127.f / amax : 0.f;
                ok &= hd[(size_t) r * K / 32 + b] == d;
                for (int i = 0; i < 32; ++i) ok &= hq[(size_t) r * K + b * 32 + i] == (int8_t) std::nearbyint(x[(size_t) r * K + b * 32 + i] * inv);
            }
        check(ok, "df2_quant_rows == CPU (scales and int8, exact)");
    }
    // attention against a double-precision CPU reference
    {
        Drafter* Dp = new Drafter(); Drafter& D = *Dp;
        const int POS = 700, n_new = 3;
        D.kring[0] = make_f32((size_t) RING * NKV * HD, 2.f); D.vring[0] = make_f32((size_t) RING * NKV * HD, 2.f);
        D.tag = dev<int64_t>(RING); D.part = dev<float>((size_t) kDf2PartChunks * 8 * 32 * 130);
        std::vector<int64_t> t(RING, -1); for (int p = 0; p < POS + n_new; ++p) t[p % RING] = ((int64_t) 1 << 32) | p; up(D.tag, t);
        std::vector<float> q((size_t) 8 * 32 * HD), kb((size_t) 8 * NKV * HD), vb(kb.size());
        for (auto& v : q) v = frand() * 2; for (auto& v : kb) v = frand() * 2; for (auto& v : vb) v = frand() * 2;
        float *dq = dev<float>(q.size()), *dk = dev<float>(kb.size()), *dv = dev<float>(vb.size()); up(dq, q); up(dk, kb); up(dv, vb);
        int32_t* ap = dev<int32_t>(1); int32_t sv = POS + n_new; Q->memcpy(ap, &sv, 4).wait(); int32_t* ep = dev<int32_t>(1); int32_t e1 = 1; Q->memcpy(ep, &e1, 4).wait();
        int8_t* xq = dev<int8_t>(8 * 32 * HD); float* xd = dev<float>(8 * 32 * HD / 32);
      for (int avk : {1064, 2064, 1128, 2128}) {
        const int av = avk / 1000, ck = avk % 1000;
        df2_attention_variant(av); df2_attention_chunk(ck);
        df2_attention(dq, D.kring[0], D.vring[0], dk, dv, ap, RING, WINDOW, D.part, xq, xd, Q, D.tag, ep); Q->wait();
        auto kr = down(D.kring[0], (size_t) RING * NKV * HD); auto vr = down(D.vring[0], (size_t) RING * NKV * HD);
        auto hq = down(xq, (size_t) 8 * 32 * HD); auto hd = down(xd, (size_t) 8 * 32 * HD / 32);
        double maxerr = 0, maxref = 0;
        const int s = POS + n_new, lo = std::max(0, s - (WINDOW - 1));
        for (int row = 0; row < 8; ++row)
            for (int hh = 0; hh < 32; ++hh) {
                const int kvh = hh / 4;
                std::vector<double> sc; std::vector<const float*> vp;
                for (int j = lo; j < s; ++j) { if (j <= s + row - WINDOW) continue; double a = 0; const float* kp = &kr[((size_t) kvh * RING + j % RING) * HD]; for (int d = 0; d < HD; ++d) a += (double) q[((size_t) row * 32 + hh) * HD + d] * kp[d]; sc.push_back(a / std::sqrt((double) HD)); vp.push_back(&vr[((size_t) kvh * RING + j % RING) * HD]); }
                for (int j = 0; j < 8; ++j) { double a = 0; const float* kp = &kb[((size_t) j * NKV + kvh) * HD]; for (int d = 0; d < HD; ++d) a += (double) q[((size_t) row * 32 + hh) * HD + d] * kp[d]; sc.push_back(a / std::sqrt((double) HD)); vp.push_back(&vb[((size_t) j * NKV + kvh) * HD]); }
                double m = -1e300; for (double v : sc) m = std::max(m, v);
                double Z = 0; std::vector<double> o(HD, 0.0);
                for (size_t j = 0; j < sc.size(); ++j) { const double w = std::exp(sc[j] - m); Z += w; for (int d = 0; d < HD; ++d) o[d] += w * vp[j][d]; }
                for (int d = 0; d < HD; ++d) {
                    const double ref = o[d] / Z;
                    const size_t blk = (size_t) hh * (HD / 32) + d / 32;   // (kvh * 4 + g) * 4 + d / 32
                    const double got = (double) hq[(size_t) row * 32 * HD + blk * 32 + d % 32] * hd[(size_t) row * 32 * HD / 32 + blk];
                    maxerr = std::max(maxerr, std::fabs(got - ref)); maxref = std::max(maxref, std::fabs(ref));
                }
            }
        std::printf("  attention: max |got - ref| = %.3e (max |ref| = %.3e; int8 quantization step ~ %.1e)\n", maxerr, maxref, maxref / 127.0);
        { char b2[96]; std::snprintf(b2, sizeof b2, "df2_attention v%d chunk %d (ring 700 + 3, tags, window) == CPU double within the int8 step", av, ck); check(maxerr < 2.0 * maxref / 127.0, b2); }
      }
      df2_attention_variant(0); df2_attention_chunk(0);
    }
    prefix_test();
    std::printf("%s\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED");
    return g_fail ? 1 : 0;
}

// ---------------------------------------------------------------- bench
struct Stats { double med, p5, p95, mean; };
Stats stats(const std::vector<double>& v) { double s = 0; for (double x : v) s += x; return Stats{med(v), pct(v, 0.05), pct(v, 0.95), s / (double) v.size()}; }

int run_bench() {
    const int POS = envi("DF2B_POS", 600), ITERS = envi("DF2B_ITERS", 100), NNEW = envi("DF2B_NNEW", 3);
    const int standin_mb = envi("DF2B_STANDIN_MB", 0);
    const double gap = envd("DF2B_GAPMS", 0.0);
    std::printf("device: %s  max wg %zu  mem %.1f GiB\n", Q->get_device().get_info<sycl::info::device::name>().c_str(),
                (size_t) Q->get_device().get_info<sycl::info::device::max_work_group_size>(),
                (double) Q->get_device().get_info<sycl::info::device::global_mem_size>() / (1024.0 * 1024 * 1024));
    Drafter D;
    D.nd = envi("DF2B_ND", 7);
    build(D);
    reset_ring(D, POS);
    float* standin = nullptr; size_t standin_n = (size_t) standin_mb * 1024 * 1024 / 4; float* sink = nullptr;
    if (standin_mb) { standin = make_f32(standin_n, 1.f); sink = dev<float>(1024); }
    auto run_standin = [&] {
        if (!standin) return;
        const float* s = standin; float* sk = sink; const size_t n = standin_n;
        Q->parallel_for(sycl::nd_range<1>(1024 * 256, 256), [=](sycl::nd_item<1> it) {
            float a = 0; for (size_t i = it.get_global_id(0); i < n; i += 1024 * 256) a += s[i];
            if (a == 12345.678f) sk[0] = a;
        });
    };
    double vram = 0;
    std::printf("config: ring pos %d, n_new %d, drafts %d, iters %d, stand-in %d MB, gap %.2f ms\n", POS, NNEW, D.nd, ITERS, standin_mb, gap);
    (void) vram;
    // warm up (JIT every kernel)
    set_meta(D, NNEW, POS, 5);
    for (int i = 0; i < 6; ++i) { set_meta(D, NNEW, POS, 5); record_round(D, NNEW); Q->wait(); }
    {   // finite-ness of the round's data (random weights could overflow; NaN / Inf would also slow special-value paths)
        auto stat = [&](const char* nm, const float* d, size_t n) {
            auto h = down(d, n); size_t bad = 0; double mx = 0;
            for (float v : h) { if (!std::isfinite(v)) ++bad; else mx = std::max(mx, (double) std::fabs(v)); }
            std::printf("data %-8s: %zu values, %zu not finite, max |v| %.3g\n", nm, n, bad, mx);
        };
        stat("h", D.h, (size_t) B * N); stat("fn", D.fn, (size_t) B * N); stat("dyn", D.dyn, (size_t) B * 640); stat("qv", D.qv, (size_t) B * NH * HD);
        stat("ao-ish o", D.o, (size_t) B * N); stat("gate", D.gate, (size_t) B * I); stat("logits", D.logits, (size_t) D.nd * NDV);
    }
    uint64_t cs = checksum(D.h_out, 64);
    std::printf("draft ids: "); for (int i = 0; i < 7; ++i) std::printf("%d ", D.h_out[i]); std::printf("  probs: "); for (int i = 0; i < 7; ++i) std::printf("%.4f ", D.h_prob[i]); std::printf("  checksum %016llx\n", (unsigned long long) cs);

    // (1) direct launches
    std::vector<double> wall;
    for (int i = 0; i < ITERS + 10; ++i) {
        run_standin();
        if (gap > 0) { Q->wait(); std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(gap)); }
        Q->wait();
        const auto t0 = Clock::now();
        set_meta(D, NNEW, POS, 5);
        record_round(D, NNEW);
        Q->wait();
        const double w = ms_since(t0);
        if (i >= 10) wall.push_back(w);
    }
    const Stats sd = stats(wall);
    std::printf("DIRECT   wall ms/round: median %.3f  p5 %.3f  p95 %.3f  mean %.3f\n", sd.med, sd.p5, sd.p95, sd.mean);
    // host submit time alone
    {
        std::vector<double> sub;
        for (int i = 0; i < 30; ++i) { Q->wait(); const auto t0 = Clock::now(); set_meta(D, NNEW, POS, 5); record_round(D, NNEW); sub.push_back(ms_since(t0)); Q->wait(); }
        std::printf("DIRECT   host submit ms (until the last launch returns): median %.3f\n", med(sub));
    }
    // (2) graph replay
    if (envi("DF2B_GRAPH", 1)) {
        try {
            sx::command_graph<sx::graph_state::modifiable> g(Q->get_context(), Q->get_device());
            g.begin_recording(*Q);
            record_round(D, NNEW, false);   // compute only: the meta copy and the result copy stay outside the graph
            g.end_recording();
            auto exec = g.finalize();
            std::vector<double> gw;
            for (int i = 0; i < ITERS + 10; ++i) {
                run_standin();
                if (gap > 0) { Q->wait(); std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(gap)); }
                Q->wait();
                const auto t0 = Clock::now();
                set_meta(D, NNEW, POS, 5);
                Q->memcpy(D.d_meta, D.h_meta, (32 + 2048) * 4);
                Q->ext_oneapi_graph(exec);
                Q->memcpy(D.h_out, D.d_out, 64 * 4);
                Q->wait();
                if (i >= 10) gw.push_back(ms_since(t0));
            }
            const Stats sg = stats(gw);
            std::printf("GRAPH    wall ms/round: median %.3f  p5 %.3f  p95 %.3f  mean %.3f   checksum %016llx\n", sg.med, sg.p5, sg.p95, sg.mean, (unsigned long long) checksum(D.h_out, 64));
        } catch (const std::exception& e) {
            std::printf("GRAPH    unavailable: %s\n", e.what());
        }
    }
    // (3) per-kernel profile (markers after each launch)
    if (envi("DF2B_PROFILE", 1)) {
        D.profile = true;
        std::vector<std::pair<std::string, double>> tot; int nrun = 0;
        for (int i = 0; i < 20; ++i) {
            D.marks.clear();
            Q->wait();
            sycl::event e0 = Q->single_task([] {});
            D.marks.emplace_back("start", e0);
            set_meta(D, NNEW, POS, 5);
            record_round(D, NNEW);
            Q->wait();
            if (i < 3) continue;
            ++nrun;
            for (size_t k = 1; k < D.marks.size(); ++k) {
                const double a = (double) D.marks[k - 1].second.get_profiling_info<sycl::info::event_profiling::command_end>();
                const double b = (double) D.marks[k].second.get_profiling_info<sycl::info::event_profiling::command_end>();
                const double ms = (b - a) * 1e-6;
                auto it = std::find_if(tot.begin(), tot.end(), [&](auto& t) { return t.first == D.marks[k].first; });
                if (it == tot.end()) tot.emplace_back(D.marks[k].first, ms); else it->second += ms;
            }
        }
        D.profile = false;
        double sum = 0; for (auto& t : tot) sum += t.second;
        std::printf("PROFILE (marker after every launch, so each includes its launch gap; ms per round, sum %.3f over %d rounds):\n", sum / nrun, nrun);
        std::sort(tot.begin(), tot.end(), [](auto& a, auto& b) { return a.second > b.second; });
        for (auto& t : tot) std::printf("  %-16s %7.3f\n", t.first.c_str(), t.second / nrun);
    }
    return 0;
}

// ---------------------------------------------------------------- per-kernel micro timings
template <class F> double time_us(F&& f, int iters) {
    for (int i = 0; i < 5; ++i) f();
    Q->wait();
    const auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) f();
    Q->wait();
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / iters;
}

int run_gemm_bench() {
    if (envi("DF2B_HOSTALLOC", 0) == 1) (void) strata::host_malloc_polled(4096, *Q);   // uncached host memory (zeMemAllocHost), as the engine's doorbells
    if (envi("DF2B_HOSTALLOC", 0) == 2) (void) sycl::malloc_host(4096, *Q);
    if (envi("DF2B_BIGALLOC_MB", 0)) { void* bp = sycl::malloc_device((size_t) envi("DF2B_BIGALLOC_MB", 0) << 20, *Q); Q->memset(bp, 1, (size_t) envi("DF2B_BIGALLOC_MB", 0) << 20).wait(); }   // touched
    for (int i = 0; i < envi("DF2B_NALLOC", 0); ++i) (void) sycl::malloc_device(4096 + 64 * i, *Q);
    for (int i = 0; i < envi("DF2B_NMED", 0); ++i) { void* mp = sycl::malloc_device(8 << 20, *Q); Q->memset(mp, 1, 8 << 20); }
    Q->wait();
    if (const char* pre = std::getenv("DF2B_PRE")) {
        const std::string ps = pre;
        if (ps.find("rope") != std::string::npos) { double* r = dev<double>(64); df2_rope_inv(10000.0f, r, Q); Q->wait(); }
        if (ps.find("head") != std::string::npos) (void) make_head();
        if (ps.find("cb") != std::string::npos) { std::vector<uint16_t> cb((size_t) VOCAB * RANK, 0x3c00); uint16_t* d = dev<uint16_t>(cb.size()); up(d, cb); }
        if (ps.find("ring") != std::string::npos) for (int l = 0; l < 10; ++l) (void) make_f32((size_t) RING * NKV * HD, 2.f);
        if (ps.find("layers") != std::string::npos) for (int l = 0; l < 5; ++l) { make_mat(I, N); make_mat(I, N); make_mat(N, I); make_mat(4096, N); }
        if (ps.find("host") != std::string::npos) { (void) strata::host_malloc_polled(128, *Q); (void) strata::host_malloc_polled(256, *Q); }
        Q->wait();
    }
    std::printf("device: %s\n", Q->get_device().get_info<sycl::info::device::name>().c_str());
    struct Shape { const char* name; int n_out, n_in, nmat; };
    const Shape shapes[] = {{"fc", N, 5 * N, 1}, {"qkv(3)", NH * HD, N, 3}, {"o", N, NH * HD, 1}, {"gate+up(2)", I, N, 2}, {"down", N, I, 1}, {"akp/mkp", 640, N, 1}, {"hsel", RANK, N, 1}};
    int vs[16], nv = 0;
    if (const char* e = std::getenv("DF2B_VARIANTS")) { for (const char* c = e; *c;) { vs[nv++] = std::atoi(c); while (*c && *c != ',') ++c; if (*c == ',') ++c; } }
    else { const int d[] = {28, 12, 40, 41, 42, 43}; for (int v : d) vs[nv++] = v; }
    const int ncs[] = {8, 4, 1};
    std::printf("us per launch (200 back-to-back), GB/s of weights+scales; columns 8 / 4 / 1\n%-12s %8s", "shape", "MB");
    for (int i = 0; i < nv; ++i) std::printf(" | v%-2d 8col  4col  1col", vs[i]);
    std::printf("\n");
    const int ncopy = envi("DF2B_COLD", 0) ? 8 : 1;   // cold: the launches rotate over 8 copies of the matrices (> L2), as a round's 5 layers do
    for (const Shape& s : shapes) {
        std::vector<std::vector<Mat4>> mats(ncopy); std::vector<float*> ys;
        for (int k = 0; k < ncopy; ++k)
            for (int m = 0; m < s.nmat; ++m) { mats[k].push_back(make_mat(s.n_out, s.n_in)); if (k == 0) ys.push_back(dev<float>((size_t) 8 * s.n_out)); }
        int8_t* xq = dev<int8_t>((size_t) 8 * s.n_in); float* xd = dev<float>((size_t) 8 * s.n_in / 32);
        {   // random activations (the bit toggling of real data), scales 0.01
            std::vector<int8_t> hq((size_t) 8 * s.n_in); for (auto& v : hq) v = (int8_t) rnd();
            std::vector<float> hd((size_t) 8 * s.n_in / 32, 0.01f);
            Q->memcpy(xq, hq.data(), hq.size()).wait(); Q->memcpy(xd, hd.data(), hd.size() * 4).wait();
        }
        const double mb = (double) s.nmat * s.n_out * s.n_in * 0.5625 / 1e6;
        std::printf("%-12s %8.1f", s.name, mb);
        for (int i = 0; i < nv; ++i) {
            df2_gemm_variant(vs[i]);
            std::printf(" |     ");
            for (int nc : ncs) {
                std::vector<std::vector<Df2Mat>> dm(ncopy);
                for (int k = 0; k < ncopy; ++k) for (int m = 0; m < s.nmat; ++m) dm[k].push_back(Df2Mat{mats[k][m].w, ys[m], s.n_out, true});
                int it = 0;
                const double us = time_us([&] { df2_gemm(dm[it++ % ncopy].data(), s.nmat, xq, xd, s.n_in, nc, Q); }, 200);
                std::printf(" %5.0f(%3.0f)", us, mb * 1e3 / us);
            }
        }
        std::printf("\n");
        for (auto& v : mats) for (auto& m : v) sycl::free(m.w, *Q);
        for (auto y : ys) sycl::free(y, *Q);
        sycl::free(xq, *Q); sycl::free(xd, *Q);
    }
    df2_gemm_variant(0);
    return 0;
}

int run_attn_bench() {
    std::printf("device: %s\n", Q->get_device().get_info<sycl::info::device::name>().c_str());
    float* kr = make_f32((size_t) RING * NKV * HD, 2.f); float* vr = make_f32((size_t) RING * NKV * HD, 2.f);
    int64_t* tag = dev<int64_t>(RING); float* part = dev<float>((size_t) kDf2PartChunks * 8 * 32 * 130);
    float* q = make_f32((size_t) 8 * 32 * HD, 2.f); float* kb = make_f32((size_t) 8 * NKV * HD, 2.f); float* vb = make_f32((size_t) 8 * NKV * HD, 2.f);
    int32_t* ap = dev<int32_t>(1); int32_t* ep = dev<int32_t>(1); int32_t e1 = 1; Q->memcpy(ep, &e1, 4).wait();
    int8_t* xq = dev<int8_t>(8 * 32 * HD); float* xd = dev<float>(8 * 32 * HD / 32);
    for (int avk : {1064, 2064, 1128, 2128}) for (int pos : {100, 600, 1200, 2100}) {
        const int av = avk / 1000, ck = avk % 1000;
        df2_attention_variant(av); df2_attention_chunk(ck);
        std::vector<int64_t> t(RING, -1); for (int p = std::max(0, pos - 2047); p < pos; ++p) t[p % RING] = ((int64_t) 1 << 32) | p;
        Q->memcpy(tag, t.data(), RING * 8).wait();
        int32_t sv = pos; Q->memcpy(ap, &sv, 4).wait();
        const double us = time_us([&] { df2_attention(q, kr, vr, kb, vb, ap, RING, WINDOW, part, xq, xd, Q, tag, ep); }, 100);
        std::printf("attention v%d chunk %d, s = %d: %.0f us per layer\n", av, ck, pos, us);
    }
    return 0;
}

// ---------------------------------------------------------------- dp4a throughput
extern "C" SYCL_EXTERNAL int __builtin_IB_dp4a_ss(int c, int a, int b);
template <int MODE> int dp4a_mode(int a, int b, int c) {
    if constexpr (MODE == 0) return dpct::dp4a(a, b, c);
    else if constexpr (MODE == 1) return __builtin_IB_dp4a_ss(c, a, b);
    else {   // sycl vec of int8
        const auto va = sycl::bit_cast<sycl::vec<int8_t, 4>>(a), vb = sycl::bit_cast<sycl::vec<int8_t, 4>>(b);
        return c + (int) va[0] * vb[0] + (int) va[1] * vb[1] + (int) va[2] * vb[2] + (int) va[3] * vb[3];
    }
}
template <int MODE> double dp4a_rate(int iters) {
    const size_t items = 28 * 128 * 64;
    int* out = dev<int>(items);
    int* in = dev<int>(items); std::vector<int> h(items); for (auto& v : h) v = (int) rnd() * 2654435761u; up(in, h);
    auto run = [&] {
        Q->parallel_for(sycl::nd_range<1>(items, 64), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const size_t i = it.get_global_id(0);
            int a0 = in[i], a1 = a0 ^ 0x1234567, a2 = a0 * 3, a3 = a0 + 77, b = in[(i + 1) % items];
            int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (int k = 0; k < iters; ++k) {
                s0 = dp4a_mode<MODE>(a0, b, s0); s1 = dp4a_mode<MODE>(a1, b, s1); s2 = dp4a_mode<MODE>(a2, b, s2); s3 = dp4a_mode<MODE>(a3, b, s3);
                a0 += 0x01010101; a1 += 0x01010101; a2 += 0x01010101; a3 += 0x01010101;
            }
            out[i] = s0 + s1 + s2 + s3;
        });
    };
    run(); Q->wait();
    const auto t0 = Clock::now(); run(); Q->wait();
    const double sec = ms_since(t0) * 1e-3;
    sycl::free(out, *Q); sycl::free(in, *Q);
    return (double) items * iters * 4 / sec / 1e9;
}
int run_dp4a() {
    std::printf("dp4a throughput, Gdp4a/s (4 independent chains): dpct::dp4a %.0f   __builtin_IB_dp4a_ss %.0f   vec<int8,4> %.0f\n", dp4a_rate<0>(20000), dp4a_rate<1>(20000), dp4a_rate<2>(20000));
    return 0;
}

int run_bubble() {
    float* a = dev<float>(4096); Q->memset(a, 0, 4096 * 4).wait();
    const double us1 = time_us([&] { Q->single_task([] {}); }, 2000);
    const double us2 = time_us([&] { Q->parallel_for(sycl::nd_range<1>(256, 64), [=](sycl::nd_item<1> it) { a[it.get_global_id(0)] += 1.f; }); }, 2000);
    const double us3 = time_us([&] { Q->parallel_for(sycl::nd_range<1>(32 * 28 * 64, 64), [=](sycl::nd_item<1> it) { a[it.get_global_id(0) & 4095] += 1.f; }); }, 2000);
    std::printf("back-to-back in-order launches, us each: empty single_task %.1f   256-item kernel %.1f   57344-item kernel %.1f\n", us1, us2, us3);
    return 0;
}

int run_small() {
    Drafter D; D.nd = 7; build(D); reset_ring(D, 600);
    set_meta(D, 3, 600, 5);
    Layer& y = D.ly[0];
    {
        const double d0 = time_us([&] { gemm2(y.gate, y.up, D.gate, D.up, B, D.xqp, D.xdp); }, 200);
        std::printf("gate_up right after build (before any round): %.1f us\n", d0);
        record_round(D, 3); Q->wait();
        const double d1 = time_us([&] { gemm2(y.gate, y.up, D.gate, D.up, B, D.xqp, D.xdp); }, 200);
        std::printf("gate_up after one round: %.1f us\n", d1);
        const double d2 = time_us([&] { df2_attention(D.qn, D.kring[0], D.vring[0], D.kblk, D.vv, D.d_meta + 17, RING, WINDOW, D.part, D.xqp, D.xdp, Q, D.tag, D.d_meta + 18); }, 200);
        std::printf("attention after one round: %.1f us\n", d2);
    }
    auto t_all = [&](const char* tag) {
        const double a = time_us([&] { df2_conv_quant(D.xn, D.dyn, 640, y.base_m, B, N, GROUP, D.xqp, D.xdp, Q); }, 1000);
        const double b = time_us([&] { df2_conv_rms_quant(D.o, D.dyn + 320, 640, y.base_a + 2 * N, D.h, y.ln_post, D.xn, B, N, GROUP, EPS, D.xqp, D.xdp, Q); }, 1000);
        const double c = time_us([&] { df2_swiglu_quant(D.gate, D.up, B, I, D.xqp, D.xdp, Q); }, 1000);
        const double d = time_us([&] { gemm2(y.gate, y.up, D.gate, D.up, B, D.xqp, D.xdp); }, 200);
        const double e = time_us([&] { df2_attention(D.qn, D.kring[0], D.vring[0], D.kblk, D.vv, D.d_meta + 17, RING, WINDOW, D.part, D.xqp, D.xdp, Q, D.tag, D.d_meta + 18); }, 200);
        std::printf("%-26s conv_quant %6.1f  conv_rms_quant %6.1f  swiglu_quant %6.1f  gate_up %6.1f  attention %6.1f  (us)\n", tag, a, b, c, d, e);
    };
    t_all("in-round data");
    Q->memset(D.xn, 0, (size_t) B * N * 4); Q->memset(D.dyn, 0, (size_t) B * 640 * 4); Q->memset(D.o, 0, (size_t) B * N * 4); Q->memset(D.h, 0, (size_t) B * N * 4);
    Q->memset(D.gate, 0, (size_t) B * I * 4); Q->memset(D.up, 0, (size_t) B * I * 4); Q->memset(D.xqp, 0, (size_t) 5 * N * 8); Q->memset(D.xdp, 0, 5 * N / 32 * 8 * 4); Q->wait();
    t_all("zeroed activations");
    std::vector<float> hx((size_t) B * N); for (auto& v : hx) v = frand();
    Q->memcpy(D.xn, hx.data(), hx.size() * 4); Q->memcpy(D.o, hx.data(), hx.size() * 4); Q->memcpy(D.h, hx.data(), hx.size() * 4);
    std::vector<float> hg((size_t) B * I); for (auto& v : hg) v = frand();
    Q->memcpy(D.gate, hg.data(), hg.size() * 4); Q->memcpy(D.up, hg.data(), hg.size() * 4);
    std::vector<float> hd((size_t) B * 640); for (auto& v : hd) v = frand();
    Q->memcpy(D.dyn, hd.data(), hd.size() * 4); Q->wait();
    t_all("small random activations");
    return 0;
}

int run_poison() {
    Drafter D; D.nd = 7; build(D); reset_ring(D, 600);
    set_meta(D, 3, 600, 5);
    Q->memcpy(D.d_meta, D.h_meta, 128).wait();
    Layer& y = D.ly[0];
    auto probe = [&](const char* after) {
        const double e = time_us([&] { df2_attention(D.qn, D.kring[0], D.vring[0], D.kblk, D.vv, D.d_meta + 17, RING, WINDOW, D.part, D.xqp, D.xdp, Q, D.tag, D.d_meta + 18); }, 100);
        const double g = time_us([&] { gemm2(y.gate, y.up, D.gate, D.up, B, D.xqp, D.xdp); }, 100);
        const double c = time_us([&] { df2_conv_quant(D.xn, D.dyn, 640, y.base_m, B, N, GROUP, D.xqp, D.xdp, Q); }, 300);
        std::printf("after %-26s attention %7.1f  gate_up %7.1f  conv_quant %6.1f us\n", after, e, g, c);
    };
    probe("build");
    df2_quant_rows(D.taps, 5 * N, 3, D.xqc, D.xdc, Q); Q->wait(); probe("quant_rows");
    df2_rms_quant(D.fcy, D.hidden_norm, nullptr, 3, N, EPS, D.xqc, D.xdc, Q); Q->wait(); probe("rms_quant");
    df2_kv_ring(D.kc, D.vc, D.ly[0].kn, D.d_meta, D.rt_ctx, D.kring[0], D.vring[0], RING, 3, NKV, EPS, Q, D.tag, D.d_meta + 18); Q->wait(); probe("kv_ring");
    embed_stub(D); Q->wait(); probe("embed");
    df2_conv_rms_quant(D.o, D.dyn + 320, 640, y.base_a + 2 * N, D.h, y.ln_post, D.xn, B, N, GROUP, EPS, D.xqp, D.xdp, Q); Q->wait(); probe("conv_rms_quant");
    df2_swiglu_quant(D.gate, D.up, B, I, D.xqp, D.xdp, Q); Q->wait(); probe("swiglu_quant");
    df2_norm_rope_qk(D.qv, D.kv, y.qn, y.kn, D.rt_blk, D.qn, D.kblk, B, NH, NKV, EPS, Q); Q->wait(); probe("norm_rope_qk");
    native_quantize_q8_1(D.fn + N, D.xq_head, N, 7, Q); Q->wait(); probe("native_quantize_q8_1");
    native_mmvq(GGML_Q8_0, D.head_w, D.xq_head, D.logits, N, NDV, 7, Q); Q->wait(); probe("native_mmvq head");
    df2_topk(D.logits, NDV, D.dvocab, 7, D.topscratch, D.topval, D.toptok, Q); Q->wait(); probe("topk");
    df2_select(D.hsel_out, D.topval, D.toptok, D.pred_cb, D.succ_cb, RANK, 7, D.d_meta + 16, D.d_out, (float*) D.d_out + 16, Q); Q->wait(); probe("select");
    return 0;
}

int run_poison2() {
    Drafter D; D.nd = 7; build(D); reset_ring(D, 600);
    Layer& y = D.ly[0];
    auto probe = [&](const char* after) {
        const double e = time_us([&] { df2_attention(D.qn, D.kring[0], D.vring[0], D.kblk, D.vv, D.d_meta + 17, RING, WINDOW, D.part, D.xqp, D.xdp, Q, D.tag, D.d_meta + 18); }, 100);
        const double g = time_us([&] { gemm2(y.gate, y.up, D.gate, D.up, B, D.xqp, D.xdp); }, 100);
        const double c = time_us([&] { df2_conv_quant(D.xn, D.dyn, 640, y.base_m, B, N, GROUP, D.xqp, D.xdp, Q); }, 300);
        std::printf("after %-44s attention %7.1f  gate_up %7.1f  conv_quant %6.1f us\n", after, e, g, c);
    };
    const int mode = envi("DF2B_P2", 0);
    probe("nothing");
    int32_t* hp = (int32_t*) strata::host_malloc_polled(4096, *Q);   // uncached host
    int32_t* hc = (int32_t*) sycl::malloc_host(4096, *Q);             // cached host
    int32_t* sh = (int32_t*) sycl::malloc_shared(4096, *Q);
    int32_t* dv = dev<int32_t>(1024);
    std::memset(hp, 0, 4096); std::memset(hc, 0, 4096); Q->memset(dv, 0, 4096).wait();
    if (mode == 0 || mode == 1) { Q->parallel_for(sycl::range<1>(64), [=](sycl::id<1> i) { dv[i] = hp[i]; }).wait(); probe("kernel READ of uncached host USM"); }
    if (mode == 0 || mode == 2) { Q->parallel_for(sycl::range<1>(64), [=](sycl::id<1> i) { dv[i] = hc[i]; }).wait(); probe("kernel READ of cached host USM"); }
    if (mode == 0 || mode == 3) { Q->parallel_for(sycl::range<1>(64), [=](sycl::id<1> i) { hp[i] = (int) i; }).wait(); probe("kernel WRITE of uncached host USM"); }
    if (mode == 0 || mode == 4) { Q->parallel_for(sycl::range<1>(64), [=](sycl::id<1> i) { hc[i] = (int) i; }).wait(); probe("kernel WRITE of cached host USM"); }
    if (mode == 0 || mode == 5) { Q->parallel_for(sycl::range<1>(64), [=](sycl::id<1> i) { sh[i] = (int) i; }).wait(); probe("kernel WRITE of shared USM"); }
    return 0;
}

// ---------------------------------------------------------------- the prompt's context update (the drafter's tap rows of the last 2047 prompt positions)
int run_ctx() {
    Drafter D; D.nd = 7; build(D); reset_ring(D, 0);
    const int NCH = envi("DF2B_CHUNKS", 256), ROWS = 8;
    int8_t* hq = (int8_t*) sycl::malloc_host((size_t) ROWS * N, *Q);
    float* hd = (float*) sycl::malloc_host((size_t) ROWS * (N / 32) * 4, *Q);
    std::vector<int32_t*> metas;   // per chunk, its own host staging (positions + rope tables), for the modes that do not wait between chunks
    for (int k = 0; k < NCH; ++k) { set_meta(D, ROWS, (int64_t) k * ROWS, 0); int32_t* m = (int32_t*) sycl::malloc_host((32 + 2048) * 4, *Q); std::memcpy(m, D.h_meta, (32 + 2048) * 4); metas.push_back(m); }
    auto warm = [&] { for (int i = 0; i < 4; ++i) { set_meta(D, ROWS, 0, 0); Q->memcpy(D.d_meta, D.h_meta, (32 + 2048) * 4); record_context(D, ROWS); } Q->wait(); };
    warm();
    auto per = [&](const char* name, double total_ms, int nch) { std::printf("  %-62s %8.1f ms total, %6.3f ms per 8-row chunk (%d chunks = %d rows)\n", name, total_ms, total_ms / nch, nch, nch * ROWS); };
    std::printf("context update of %d rows in chunks of %d, the Q4_0 pack's shapes (fc 12800 -> 2560, K/V of 5 layers):\n", NCH * ROWS, ROWS);
    { // (a) as Dflash2Drafter::context does it: host staging + meta copy + the update's kernels + the log's copy to the host + a wait, per chunk
        const auto t0 = Clock::now();
        for (int k = 0; k < NCH; ++k) {
            set_meta(D, ROWS, (int64_t) k * ROWS, 0);
            Q->memcpy(D.d_meta, D.h_meta, (32 + 2048) * 4);
            record_context(D, ROWS);
            Q->memcpy(hq, D.xqc, (size_t) ROWS * N); Q->memcpy(hd, D.xdc, (size_t) ROWS * (N / 32) * 4);
            Q->wait();
        }
        per("(a) the engine's loop: stage, 13 kernels, log copy, wait per chunk", ms_since(t0), NCH);
    }
    { // (b) host staging alone
        const auto t0 = Clock::now();
        for (int k = 0; k < NCH; ++k) set_meta(D, ROWS, (int64_t) k * ROWS, 0);
        per("(b) the host staging alone (positions + 16 rope tables)", ms_since(t0), NCH);
    }
    { // (c) the kernels alone, queued back to back, one wait
        const auto t0 = Clock::now();
        for (int k = 0; k < NCH; ++k) record_context(D, ROWS);
        Q->wait();
        per("(c) the 13 kernels alone, queued back to back, one wait", ms_since(t0), NCH);
    }
    { // (d) staging copies + kernels queued, no wait per chunk (each chunk's staging from its own buffer)
        const auto t0 = Clock::now();
        for (int k = 0; k < NCH; ++k) { Q->memcpy(D.d_meta, metas[(size_t) k], (32 + 2048) * 4); record_context(D, ROWS); }
        Q->wait();
        per("(d) meta copy + 13 kernels per chunk, pre-staged, one wait", ms_since(t0), NCH);
    }
    { // (e) one chunk, a marker after every launch: GPU ms per kernel of the update
        D.profile = true;
        std::vector<std::pair<std::string, double>> tot; int nrun = 0;
        for (int i = 0; i < 12; ++i) {
            D.marks.clear(); Q->wait();
            D.marks.emplace_back("start", Q->single_task([] {}));
            set_meta(D, ROWS, 0, 0); Q->memcpy(D.d_meta, D.h_meta, (32 + 2048) * 4); record_context(D, ROWS); Q->wait();
            if (i < 2) continue;
            ++nrun;
            for (size_t k = 1; k < D.marks.size(); ++k) {
                const double a = (double) D.marks[k - 1].second.get_profiling_info<sycl::info::event_profiling::command_end>();
                const double b = (double) D.marks[k].second.get_profiling_info<sycl::info::event_profiling::command_end>();
                auto it = std::find_if(tot.begin(), tot.end(), [&](auto& t) { return t.first == D.marks[k].first; });
                if (it == tot.end()) tot.emplace_back(D.marks[k].first, (b - a) * 1e-6); else it->second += (b - a) * 1e-6;
            }
        }
        D.profile = false;
        double sum = 0; for (auto& t : tot) sum += t.second;
        std::printf("  (e) one 8-row chunk, a marker after every launch (each includes its launch gap), ms: total %.3f over %d launches:", sum / nrun, (int) D.marks.size() - 1);
        for (auto& t : tot) std::printf(" %s %.3f", t.first.c_str(), t.second / nrun);
        std::printf("\n");
    }
    { // (f) weight bytes read per 8-row chunk and the bandwidth the best case would need
        const double mb = (18.4 + 5 * 3.54);
        std::printf("  (f) weight bytes per chunk: fc 18.4 MB + 5 x (k,v) 3.5 MB = %.1f MB; for %d chunks %.2f GB; at 400 GB/s that is %.1f ms per 8-row chunk\n", mb, NCH, mb * NCH / 1000.0, mb / 400.0);
    }
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (const char* d = std::getenv("DF2B_DUP")) g_dup = d;
    g_dupn = envi("DF2B_DUPN", 1);
    g_host_out = envi("DF2B_HOSTOUT", 0); g_meta_kernel = envi("DF2B_METAKERNEL", 0);
    static sycl::queue q(sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    Q = &q;
    const std::string mode = argc > 1 ? argv[1] : "bench";
    return mode == "test" ? run_tests() : mode == "gemm" ? run_gemm_bench() : mode == "attn" ? run_attn_bench() : mode == "dp4a" ? run_dp4a() : mode == "bubble" ? run_bubble() : mode == "small" ? run_small() : mode == "poison" ? run_poison() : mode == "poison2" ? run_poison2() : mode == "ctx" ? run_ctx() : run_bench();
}
