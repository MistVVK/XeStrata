// tools/vision/own/vit_cpu.cpp - see vit_cpu.h
#include "vit_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace strata_vit {

namespace {

constexpr int kKC = 256;        // the k block of a matrix product: a packed weight panel is kKC x (2 tiles) floats
constexpr int kQB = 48;         // query rows of an attention task (a multiple of every tier's MR)
constexpr int kTK = 256;        // keys per attention tile
constexpr int kRowsChunk = 4096;   // rows of the activations that are live at once outside the attention

struct Mat {                     // a weight matrix [rows = out][cols = in]
    std::vector<uint16_t> pk;    // BF16 / F16 matrices as 16-bit panels [ceil(rows / NR)][cols][NR] (the rows past `rows` zero), made at load; the stored copy is not kept
    std::vector<float> f32;      // anything else (F32, Q8_0 expanded)
    int kind = 0;                // 0 float32, 1 BF16, 2 F16
    int rows = 0, cols = 0;
    const void* data() const { return f32.data(); }       // kind 0 only (the 16-bit matrices live in pk)
};

struct Layer {
    std::vector<float> ln1_w, ln1_b, ln2_w, ln2_b, qkv_b, o_b, up_b, down_b;
    Mat qkv, o, up, down;
};

}  // namespace

struct VitCpu::Impl {
    VitGeometry g;
    const CpuOps* ops = nullptr;
    int act_round = 0;                  // STRATA_VIT_REF_ROUND: 0 none, 1 BF16, 2 F16 (the weight's type)
    std::vector<float> patch_b, pos, post_w, post_b, mm0_b, mm2_b;
    Mat patch_w, mm0, mm2;
    std::vector<Layer> layers;
    std::unique_ptr<Team> team;
    int team_threads = 0;

    bool read_f(const Gguf& gg, const std::string& name, std::vector<float>& dst, std::string& err) {
        const GgufTensorInfo* t = gg.tensor(name);
        if (!t) { err = "the mmproj file lacks the tensor " + name; return false; }
        dst.resize((size_t) t->nelem());
        return gg.read_f32(*t, dst.data(), err);
    }
    bool read_m(const Gguf& gg, const std::string& name, Mat& m, std::string& err) {
        const GgufTensorInfo* t = gg.tensor(name);
        if (!t || t->ne.size() != 2) { err = "the mmproj file lacks the matrix " + name; return false; }
        m.cols = (int) t->ne[0];
        m.rows = (int) t->ne[1];
        const size_t n = (size_t) t->nelem();
        if (t->type == GT_F16 || t->type == GT_BF16) {
            m.kind = t->type == GT_BF16 ? 1 : 2;
            std::vector<uint16_t> raw(n);
            if (!gg.read_raw(*t, raw.data(), err)) return false;
            // panels of NR rows, k-major inside a panel: a product task converts a [kc][NR] slice to float with one contiguous pass
            const int NR = ops->NR;
            const size_t ptiles = ((size_t) m.rows + NR - 1) / NR;
            m.pk.assign(ptiles * (size_t) m.cols * NR, 0);
            for (int r = 0; r < m.rows; ++r) {
                const uint16_t* src = &raw[(size_t) r * m.cols];
                uint16_t* dst = &m.pk[(size_t) (r / NR) * m.cols * NR + (size_t) (r % NR)];
                for (int k = 0; k < m.cols; ++k) dst[(size_t) k * NR] = src[k];
            }
            return true;
        }
        m.kind = 0;
        m.f32.resize(n);
        return gg.read_f32(*t, m.f32.data(), err);
    }

    template <class F>
    void run(int n, F f) { team->run(n, [&](int task, int thr) { f(task, thr); }); }

    // rows of simple per-row work, 16 rows per task
    template <class F>
    void rows_par(int n, F f) {
        const int per = 16, tasks = (n + per - 1) / per;
        team->run(tasks, [&](int task, int) {
            const int r1 = std::min(n, (task + 1) * per);
            for (int r = task * per; r < r1; ++r) f(r);
        });
    }

    void round_inplace(float* p, size_t n, int kind) {
        if (kind == 0) return;
        const size_t blk = 16384;
        const int tasks = (int) ((n + blk - 1) / blk);
        team->run(tasks, [&](int task, int) {
            const size_t i0 = (size_t) task * blk, i1 = std::min(n, i0 + blk);
            ops->round16(p + i0, i1 - i0, kind);
        });
    }

    // C[M][w.rows] (init-mode, see CpuOps::gemm) = A[M][w.cols] * w^T (+ bias), optionally GELU on the result
    void linear(const Mat& w, const float* A, size_t lda, int M, float* C, size_t ldc, int init, const float* bias, bool gelu, std::vector<std::vector<float>>& scratch) {
        const int NR = ops->NR, PW = 2 * NR, K = w.cols, N = w.rows;
        const int npan = (N + PW - 1) / PW;
        const int target = 4 * team->size();
        int mslabs = std::max(1, std::min((target + npan - 1) / npan, (M + 47) / 48));
        const int slab = (((M + mslabs - 1) / mslabs) + ops->MR - 1) / ops->MR * ops->MR;
        mslabs = (M + slab - 1) / slab;
        team->run(npan * mslabs, [&](int task, int thr) {
            const int p = task % npan, s = task / npan;
            const int m0 = s * slab, mr = std::min(slab, M - m0);
            const int n0 = p * PW, nv = std::min(PW, N - n0);
            float* buf = scratch[(size_t) thr].data();
            for (int k0 = 0; k0 < K; k0 += kKC) {
                const int kc = std::min(kKC, K - k0);
                for (int t = 0; t < 2; ++t) {
                    const int tv = std::max(0, std::min(NR, nv - t * NR));
                    if (tv <= 0) continue;
                    if (w.kind == 0) ops->pack_w(w.data(), 0, (size_t) K, n0 + t * NR, tv, k0, kc, buf + (size_t) t * kKC * NR);
                    else ops->conv(&w.pk[((size_t) (p * 2 + t) * K + (size_t) k0) * NR], w.kind, (size_t) kc * NR, buf + (size_t) t * kKC * NR);
                }
                for (int t = 0; t < 2; ++t) {
                    const int tv = std::max(0, std::min(NR, nv - t * NR));
                    if (tv <= 0) continue;
                    const int col = n0 + t * NR;
                    ops->gemm(A + (size_t) m0 * lda + k0, lda, mr, buf + (size_t) t * kKC * NR, (size_t) NR, kc, tv, C + (size_t) m0 * ldc + col, ldc,
                              k0 == 0 ? init : 1, bias ? bias + col : nullptr);
                }
            }
            if (gelu) ops->gelu(C + (size_t) m0 * ldc + n0, ldc, mr, nv);
        });
    }
};

VitCpu::VitCpu() : impl_(new Impl()) {}
VitCpu::~VitCpu() = default;
const VitGeometry& VitCpu::geometry() const { return impl_->g; }
const char* VitCpu::isa_name() const { return impl_->ops ? impl_->ops->name : "none"; }

bool VitCpu::load(const Gguf& gg, const VitGeometry& g, std::string& err) {
    Impl& m = *impl_;
    std::string why;
    if (!vit_geometry_supported(g, why)) { err = "this tower is not supported: " + why; return false; }
    m.g = g;
    m.ops = &cpu_ops(quant::best_isa());
    // the patch embedding: the two temporal taps summed (a still picture is fed to both), kept float32 (a small matrix)
    {
        const GgufTensorInfo* t0 = gg.tensor("v.patch_embd.weight");
        if (!t0) { err = "the mmproj file lacks v.patch_embd.weight"; return false; }
        std::vector<float> w((size_t) t0->nelem());
        if (!gg.read_f32(*t0, w.data(), err)) return false;
        if (w.size() != (size_t) g.width * 3 * g.patch * g.patch) { err = "the patch embedding has an unexpected size"; return false; }
        if (g.patch_taps == 2) {
            const GgufTensorInfo* t1 = gg.tensor("v.patch_embd.weight.1");
            std::vector<float> w1(w.size());
            if (!t1 || !gg.read_f32(*t1, w1.data(), err)) { if (err.empty()) err = "the mmproj file lacks v.patch_embd.weight.1"; return false; }
            for (size_t i = 0; i < w.size(); ++i) w[i] += w1[i];
        }
        m.patch_w.kind = 0;
        m.patch_w.rows = g.width;
        m.patch_w.cols = 3 * g.patch * g.patch;
        m.patch_w.f32 = std::move(w);
    }
    if (!m.read_f(gg, "v.patch_embd.bias", m.patch_b, err) || !m.read_f(gg, "v.position_embd.weight", m.pos, err) || !m.read_f(gg, "v.post_ln.weight", m.post_w, err) ||
        !m.read_f(gg, "v.post_ln.bias", m.post_b, err) || !m.read_m(gg, "mm.0.weight", m.mm0, err) || !m.read_f(gg, "mm.0.bias", m.mm0_b, err) ||
        !m.read_m(gg, "mm.2.weight", m.mm2, err) || !m.read_f(gg, "mm.2.bias", m.mm2_b, err))
        return false;
    m.layers.assign((size_t) g.layers, Layer{});
    for (int il = 0; il < g.layers; ++il) {
        Layer& L = m.layers[(size_t) il];
        const std::string p = "v.blk." + std::to_string(il) + ".";
        if (!m.read_f(gg, p + "ln1.weight", L.ln1_w, err) || !m.read_f(gg, p + "ln1.bias", L.ln1_b, err) || !m.read_f(gg, p + "ln2.weight", L.ln2_w, err) ||
            !m.read_f(gg, p + "ln2.bias", L.ln2_b, err) || !m.read_m(gg, p + "attn_qkv.weight", L.qkv, err) || !m.read_f(gg, p + "attn_qkv.bias", L.qkv_b, err) ||
            !m.read_m(gg, p + "attn_out.weight", L.o, err) || !m.read_f(gg, p + "attn_out.bias", L.o_b, err) || !m.read_m(gg, p + "ffn_up.weight", L.up, err) ||
            !m.read_f(gg, p + "ffn_up.bias", L.up_b, err) || !m.read_m(gg, p + "ffn_down.weight", L.down, err) || !m.read_f(gg, p + "ffn_down.bias", L.down_b, err))
            return false;
    }
    m.act_round = 0;
    if (const char* e = std::getenv("STRATA_VIT_REF_ROUND"); e && std::atoi(e) != 0) {
        const int k = m.layers.empty() ? 0 : m.layers[0].qkv.kind;
        m.act_round = k;          // 1 BF16, 2 F16
    }
    return true;
}

bool VitCpu::encode(const PreparedImage& img, int threads, std::vector<float>& rows, int& nx, int& ny, std::string& err) {
    Impl& m = *impl_;
    const VitGeometry& g = m.g;
    if (m.layers.empty()) { err = "the tower is not loaded"; return false; }
    const int P = g.patch, D = g.width, H = g.heads, HD = g.head_dim, F = g.ffn;
    if (HD > 128 || HD % 4 != 0) { err = "this head size is not handled by the CPU tower"; return false; }
    if (img.w <= 0 || img.h <= 0 || img.w % g.align() != 0 || img.h % g.align() != 0) { err = "the picture size is not a multiple of patch * merge"; return false; }
    int nth = threads > 0 ? threads : (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency() / 2));
    if (!m.team || m.team_threads != nth) { m.team.reset(new Team(nth)); m.team_threads = nth; }
    const CpuOps& ops = *m.ops;
    const int pw = img.w / P, ph = img.h / P, N = pw * ph, nbx = pw / 2;
    nx = pw / 2;
    ny = ph / 2;
    const int R = std::min(N, kRowsChunk);

    std::vector<int> ty((size_t) N), tx((size_t) N);
    for (int t = 0; t < N; ++t) {
        const int b = t / 4, r = t % 4;
        ty[(size_t) t] = 2 * (b / nbx) + r / 2;
        tx[(size_t) t] = 2 * (b % nbx) + r % 2;
    }
    std::vector<std::vector<float>> scratch((size_t) nth, std::vector<float>((size_t) kKC * 2 * ops.NR));

    // ---- patch embedding: im2col of a chunk of rows in merge order, one product; then the position table (bilinear, align corners) ----
    const int K0 = 3 * P * P;
    std::vector<float> x((size_t) N * D);
    {
        std::vector<float> A((size_t) R * K0);
        const size_t plane = (size_t) img.w * img.h;
        for (int r0 = 0; r0 < N; r0 += R) {
            const int rn = std::min(R, N - r0);
            m.rows_par(rn, [&](int i) {
                const int t = r0 + i;
                float* a = &A[(size_t) i * K0];
                const int py = ty[(size_t) t], px = tx[(size_t) t];
                for (int c = 0; c < 3; ++c)
                    for (int ky = 0; ky < P; ++ky)
                        std::memcpy(a + (size_t) c * P * P + (size_t) ky * P, &img.chw[c * plane + (size_t) (py * P + ky) * img.w + (size_t) px * P], (size_t) P * sizeof(float));
            });
            m.linear(m.patch_w, A.data(), (size_t) K0, rn, x.data() + (size_t) r0 * D, (size_t) D, 0, m.patch_b.data(), false, scratch);
        }
    }
    {
        const int G = g.pos_grid;
        float sfx = (float) pw / G, sfy = (float) ph / G;
        if (pw > 1 && G > 1) sfx = (float) (pw - 1) / (G - 1);
        if (ph > 1 && G > 1) sfy = (float) (ph - 1) / (G - 1);
        m.rows_par(N, [&](int t) {
            const float fy = (float) ty[(size_t) t] / sfy, fx = (float) tx[(size_t) t] / sfx;
            int y0 = (int) std::floor(fy), x0 = (int) std::floor(fx), y1 = y0 + 1, x1 = x0 + 1;
            y0 = std::max(0, std::min(y0, G - 1)); y1 = std::max(0, std::min(y1, G - 1));
            x0 = std::max(0, std::min(x0, G - 1)); x1 = std::max(0, std::min(x1, G - 1));
            const float dy = std::max(0.0f, std::min(fy - (float) y0, 1.0f)), dx = std::max(0.0f, std::min(fx - (float) x0, 1.0f));
            const float* a = &m.pos[((size_t) y0 * G + x0) * D];
            const float* b = &m.pos[((size_t) y0 * G + x1) * D];
            const float* c = &m.pos[((size_t) y1 * G + x0) * D];
            const float* d = &m.pos[((size_t) y1 * G + x1) * D];
            float* xr = &x[(size_t) t * D];
            for (int i = 0; i < D; ++i) xr[i] += a[i] * (1 - dx) * (1 - dy) + b[i] * dx * (1 - dy) + c[i] * (1 - dx) * dy + d[i] * dx * dy;
        });
    }

    // ---- rotary tables: pair j of a head (elements j and j + HD/2), j < HD/4 turns with the row, the rest with the column ----
    const int half = HD / 2, quarter = HD / 4;
    std::vector<float> rc((size_t) ph * quarter), rs((size_t) ph * quarter), cc((size_t) pw * quarter), cs((size_t) pw * quarter);
    {
        const float theta_scale = std::pow(g.rope_theta, -2.0f / (float) half);
        std::vector<float> inv((size_t) quarter);
        float th = 1.0f;
        for (int j = 0; j < quarter; ++j) { inv[(size_t) j] = th; th *= theta_scale; }
        for (int p = 0; p < ph; ++p)
            for (int j = 0; j < quarter; ++j) { const float a = (float) p * inv[(size_t) j]; rc[(size_t) p * quarter + j] = std::cos(a); rs[(size_t) p * quarter + j] = std::sin(a); }
        for (int p = 0; p < pw; ++p)
            for (int j = 0; j < quarter; ++j) { const float a = (float) p * inv[(size_t) j]; cc[(size_t) p * quarter + j] = std::cos(a); cs[(size_t) p * quarter + j] = std::sin(a); }
    }
    const float kq_scale = 1.0f / std::sqrt((float) HD);

    // ---- the layers ----
    const size_t slack = 256;                                    // the product with V reads up to a tile past a head's columns
    std::vector<float> qkv((size_t) N * 3 * D + slack, 0.0f), ctx((size_t) N * D);
    const int Np = (N + 63) / 64 * 64;
    std::vector<float> KT((size_t) H * HD * Np, 0.0f);
    std::vector<float> hbuf((size_t) R * D), up((size_t) R * F);
    const int OD = (HD + 15) / 16 * 16;
    const int attn_tasks_q = (N + kQB - 1) / kQB;
    std::vector<std::vector<float>> sscr((size_t) nth, std::vector<float>((size_t) kQB * kTK + (size_t) kQB * OD + 2 * kQB));
    const size_t D3 = (size_t) 3 * D;

    for (const Layer& L : m.layers) {
        // q, k, v of every token
        for (int r0 = 0; r0 < N; r0 += R) {
            const int rn = std::min(R, N - r0);
            m.rows_par(rn, [&](int i) { ops.layernorm(x.data() + (size_t) (r0 + i) * D, hbuf.data() + (size_t) i * D, D, L.ln1_w.data(), L.ln1_b.data(), g.eps); });
            m.round_inplace(hbuf.data(), (size_t) rn * D, m.act_round);
            m.linear(L.qkv, hbuf.data(), (size_t) D, rn, qkv.data() + (size_t) r0 * D3, D3, 0, L.qkv_b.data(), false, scratch);
        }
        m.rows_par(N, [&](int t) {       // the rotary on q and k (with the attention's scale on q, after the optional 16-bit rounding)
            float* row = qkv.data() + (size_t) t * D3;
            const float* c1 = &rc[(size_t) ty[(size_t) t] * quarter];
            const float* s1 = &rs[(size_t) ty[(size_t) t] * quarter];
            const float* c2 = &cc[(size_t) tx[(size_t) t] * quarter];
            const float* s2 = &cs[(size_t) tx[(size_t) t] * quarter];
            for (int part = 0; part < 2; ++part)
                for (int hh = 0; hh < H; ++hh) {
                    float* v = row + (size_t) part * D + (size_t) hh * HD;
                    for (int j = 0; j < half; ++j) {
                        const float co = j < quarter ? c1[j] : c2[j - quarter], si = j < quarter ? s1[j] : s2[j - quarter];
                        const float a = v[j], b = v[j + half];
                        v[j] = a * co - b * si;
                        v[j + half] = a * si + b * co;
                    }
                }
            if (m.act_round) ops.round16(row, D3, 2);
            for (int i = 0; i < D; ++i) row[i] *= kq_scale;
        });
        // K transposed per head: KT[h][d][j]
        {
            const int jb = 64, njb = (N + jb - 1) / jb;
            m.team->run(H * njb, [&](int task, int) {
                const int hh = task / njb, j0 = (task % njb) * jb, j1 = std::min(N, j0 + jb);
                for (int j = j0; j < j1; ++j) {
                    const float* kr = qkv.data() + (size_t) j * D3 + D + (size_t) hh * HD;
                    float* kt = KT.data() + (size_t) hh * HD * Np + j;
                    for (int d = 0; d < HD; ++d) kt[(size_t) d * Np] = kr[d];
                }
            });
        }
        // tiled attention with an online softmax: a task is a head and a block of kQB queries
        m.team->run(H * attn_tasks_q, [&](int task, int thr) {
            const int hh = task / attn_tasks_q, q0 = (task % attn_tasks_q) * kQB, qr = std::min(kQB, N - q0);
            float* S = sscr[(size_t) thr].data();
            float* O = S + (size_t) kQB * kTK;
            float* mx = O + (size_t) kQB * OD;
            float* ls = mx + kQB;
            std::memset(O, 0, sizeof(float) * (size_t) kQB * OD);
            for (int r = 0; r < kQB; ++r) { mx[r] = -3.0e38f; ls[r] = 0.0f; }
            const float* Q = qkv.data() + (size_t) q0 * D3 + (size_t) hh * HD;
            const float* KTh = KT.data() + (size_t) hh * HD * Np;
            const float* Vh = qkv.data() + 2 * (size_t) D + (size_t) hh * HD;
            for (int j0 = 0; j0 < N; j0 += kTK) {
                const int tk = std::min(kTK, N - j0);
                ops.gemm(Q, D3, qr, KTh + j0, (size_t) Np, HD, tk, S, (size_t) kTK, 0, nullptr);
                ops.softmax_update(S, (size_t) kTK, qr, tk, mx, ls, O, (size_t) OD, HD);
                ops.gemm(S, (size_t) kTK, qr, Vh + (size_t) j0 * D3, D3, tk, HD, O, (size_t) OD, 1, nullptr);
            }
            for (int r = 0; r < qr; ++r) {
                const float inv = 1.0f / ls[r];
                float* dst = ctx.data() + (size_t) (q0 + r) * D + (size_t) hh * HD;
                for (int d = 0; d < HD; ++d) dst[d] = O[(size_t) r * OD + d] * inv;
            }
        });
        m.round_inplace(ctx.data(), ctx.size(), m.act_round);
        for (int r0 = 0; r0 < N; r0 += R) {      // x += o(ctx)
            const int rn = std::min(R, N - r0);
            m.linear(L.o, ctx.data() + (size_t) r0 * D, (size_t) D, rn, x.data() + (size_t) r0 * D, (size_t) D, 2, L.o_b.data(), false, scratch);
        }
        for (int r0 = 0; r0 < N; r0 += R) {      // x += down(gelu(up(LN2 x)))
            const int rn = std::min(R, N - r0);
            m.rows_par(rn, [&](int i) { ops.layernorm(x.data() + (size_t) (r0 + i) * D, hbuf.data() + (size_t) i * D, D, L.ln2_w.data(), L.ln2_b.data(), g.eps); });
            m.round_inplace(hbuf.data(), (size_t) rn * D, m.act_round);
            m.linear(L.up, hbuf.data(), (size_t) D, rn, up.data(), (size_t) F, 0, L.up_b.data(), true, scratch);
            m.round_inplace(up.data(), (size_t) rn * F, m.act_round);
            m.linear(L.down, up.data(), (size_t) F, rn, x.data() + (size_t) r0 * D, (size_t) D, 2, L.down_b.data(), false, scratch);
        }
    }

    // ---- post norm and the merger (4 consecutive tokens = one 4D row) ----
    std::vector<float>& h = ctx;       // the context is no longer needed: the post-norm output takes its place
    m.rows_par(N, [&](int t) { ops.layernorm(x.data() + (size_t) t * D, h.data() + (size_t) t * D, D, m.post_w.data(), m.post_b.data(), g.eps); });
    m.round_inplace(h.data(), h.size(), m.act_round);
    const int M = N / 4, Kin = 4 * D, Hid = g.merger_hidden, O = g.out_dim;
    rows.assign((size_t) M * O, 0.0f);
    const int RM = std::max(1, R / 4);
    std::vector<float> m1((size_t) std::min(M, RM) * Hid);
    for (int r0 = 0; r0 < M; r0 += RM) {
        const int rn = std::min(RM, M - r0);
        m.linear(m.mm0, h.data() + (size_t) r0 * Kin, (size_t) Kin, rn, m1.data(), (size_t) Hid, 0, m.mm0_b.data(), true, scratch);
        m.round_inplace(m1.data(), (size_t) rn * Hid, m.act_round);
        m.linear(m.mm2, m1.data(), (size_t) Hid, rn, rows.data() + (size_t) r0 * O, (size_t) O, 0, m.mm2_b.data(), false, scratch);
    }
    return true;
}

}  // namespace strata_vit
