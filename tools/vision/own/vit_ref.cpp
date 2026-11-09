// tools/vision/own/vit_ref.cpp - see vit_ref.h
#include "vit_ref.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace strata_vit {

namespace {

template <class F>
void parallel_for(int n, int nth, F f) {
    if (nth <= 1 || n <= 1) {
        for (int i = 0; i < n; ++i) f(i);
        return;
    }
    nth = std::min(nth, n);
    std::atomic<int> next{0};
    auto worker = [&]() {
        for (int i; (i = next.fetch_add(1)) < n;) f(i);
    };
    std::vector<std::thread> ts;
    ts.reserve((size_t) nth - 1);
    for (int t = 1; t < nth; ++t) ts.emplace_back(worker);
    worker();
    for (auto& t : ts) t.join();
}

// R rows of A against two rows of W: out[r][c] = dot(a_r, w_c)
template <int R>
inline void tile(const float* a, size_t lda, const float* w0, const float* w1, int K, float out[R][2]) {
    float acc[R][2][8];
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < 2; ++c)
            for (int l = 0; l < 8; ++l) acc[r][c][l] = 0.0f;
    int k = 0;
    for (; k + 8 <= K; k += 8) {
        for (int r = 0; r < R; ++r) {
            const float* ar = a + (size_t) r * lda + k;
            for (int l = 0; l < 8; ++l) {
                acc[r][0][l] += ar[l] * w0[k + l];
                acc[r][1][l] += ar[l] * w1[k + l];
            }
        }
    }
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < 2; ++c) {
            float s = ((acc[r][c][0] + acc[r][c][4]) + (acc[r][c][1] + acc[r][c][5])) + ((acc[r][c][2] + acc[r][c][6]) + (acc[r][c][3] + acc[r][c][7]));
            const float* w = c ? w1 : w0;
            for (int kk = k; kk < K; ++kk) s += a[(size_t) r * lda + kk] * w[kk];
            out[r][c] = s;
        }
}

// C[M][N] = A[M][K] * W[N][K]^T (+ bias[N]); A rows lda apart, C rows ldc apart
void gemm(const float* A, size_t lda, int M, int K, const float* W, int N, const float* bias, float* C, size_t ldc, int nth) {
    const int chunk = std::max(4, ((M + nth * 2 - 1) / (nth * 2) + 3) / 4 * 4);
    const int nchunks = (M + chunk - 1) / chunk;
    parallel_for(nchunks, nth, [&](int ci) {
        const int m0 = ci * chunk, m1 = std::min(M, m0 + chunk);
        for (int n0 = 0; n0 < N; n0 += 8) {
            const int n1 = std::min(N, n0 + 8);
            for (int n = n0; n < n1; n += 2) {
                const bool two = n + 1 < N;
                const float* w0 = W + (size_t) n * K;
                const float* w1 = two ? w0 + K : w0;
                int m = m0;
                for (; m + 4 <= m1; m += 4) {
                    float o[4][2];
                    tile<4>(A + (size_t) m * lda, lda, w0, w1, K, o);
                    for (int r = 0; r < 4; ++r) {
                        C[(size_t) (m + r) * ldc + n] = o[r][0] + (bias ? bias[n] : 0.0f);
                        if (two) C[(size_t) (m + r) * ldc + n + 1] = o[r][1] + (bias ? bias[n + 1] : 0.0f);
                    }
                }
                for (; m < m1; ++m) {
                    float o[1][2];
                    tile<1>(A + (size_t) m * lda, lda, w0, w1, K, o);
                    C[(size_t) m * ldc + n] = o[0][0] + (bias ? bias[n] : 0.0f);
                    if (two) C[(size_t) m * ldc + n + 1] = o[0][1] + (bias ? bias[n + 1] : 0.0f);
                }
            }
        }
    });
}

// y = LayerNorm(x) * w + b, per row of D, the sums in double (as ggml's norm)
void layer_norm(const float* x, float* y, int rows, int D, const float* w, const float* b, float eps, int nth) {
    parallel_for(rows, nth, [&](int r) {
        const float* xr = x + (size_t) r * D;
        double s = 0;
        for (int i = 0; i < D; ++i) s += xr[i];
        const float mean = (float) (s / D);
        double v = 0;
        for (int i = 0; i < D; ++i) { const double d = xr[i] - mean; v += d * d; }
        const float scale = 1.0f / std::sqrt((float) (v / D) + eps);
        float* yr = y + (size_t) r * D;
        for (int i = 0; i < D; ++i) yr[i] = (xr[i] - mean) * scale * w[i] + b[i];
    });
}

inline float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.79788456080286535587989211986876f * x * (1.0f + 0.044715f * x * x)));
}

void gelu_inplace(float* x, size_t n, int nth) {
    const int blocks = (int) ((n + 4095) / 4096);
    parallel_for(blocks, nth, [&](int bi) {
        const size_t i0 = (size_t) bi * 4096, i1 = std::min(n, i0 + 4096);
        for (size_t i = i0; i < i1; ++i) x[i] = gelu_tanh(x[i]);
    });
}

inline float dot(const float* a, const float* b, int n) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int i = 0;
    for (; i + 8 <= n; i += 8)
        for (int l = 0; l < 8; ++l) acc[l] += a[i + l] * b[i + l];
    float s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

// round to nearest even to BF16 / F16 and back (the values a 16-bit GEMM input has)
inline float round_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return f;
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    std::memcpy(&f, &u, 4);
    return f;
}

inline float round_f16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t sign = u & 0x80000000u;
    const float a = std::fabs(f);
    if (!(a < 65520.0f)) return f;                         // inf, nan, and what rounds to inf are left alone (not met in a tower)
    if (a < 6.103515625e-05f) {                            // subnormal half: a multiple of 2^-24
        const float q = std::nearbyint(a * 16777216.0f) / 16777216.0f;
        return sign ? -q : q;
    }
    int e;
    std::frexp(a, &e);                                     // a = m * 2^e, m in [0.5, 1): the half's ulp is 2^(e-11)
    const float ulp = std::ldexp(1.0f, e - 11);
    const float q = std::nearbyint(a / ulp) * ulp;
    return sign ? -q : q;
}

void round_inplace(float* p, size_t n, int kind, int nth) {
    if (kind == 0) return;
    const int blocks = (int) ((n + 8191) / 8192);
    parallel_for(blocks, nth, [&](int bi) {
        const size_t i0 = (size_t) bi * 8192, i1 = std::min(n, i0 + 8192);
        for (size_t i = i0; i < i1; ++i) p[i] = kind == 1 ? round_bf16(p[i]) : round_f16(p[i]);
    });
}

bool read_t(const Gguf& gg, const std::string& name, std::vector<float>& dst, std::string& err) {
    const GgufTensorInfo* t = gg.tensor(name);
    if (!t) { err = "the mmproj file lacks the tensor " + name; return false; }
    dst.resize((size_t) t->nelem());
    return gg.read_f32(*t, dst.data(), err);
}

}  // namespace

double cosine(const float* a, const float* b, size_t n) {
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < n; ++i) { ab += (double) a[i] * b[i]; aa += (double) a[i] * a[i]; bb += (double) b[i] * b[i]; }
    return aa > 0 && bb > 0 ? ab / std::sqrt(aa * bb) : 0.0;
}

bool VitRef::load(const Gguf& gg, const VitGeometry& g, std::string& err) {
    g_ = g;
    std::string why;
    if (!vit_geometry_supported(g, why)) { err = "this tower is not supported: " + why; return false; }
    // the two temporal taps of the patch embedding are summed: a still picture is fed to both
    std::vector<float> w0, w1;
    if (!read_t(gg, "v.patch_embd.weight", w0, err)) return false;
    if (g.patch_taps == 2) {
        if (!read_t(gg, "v.patch_embd.weight.1", w1, err)) return false;
        for (size_t i = 0; i < w0.size(); ++i) w0[i] += w1[i];
    }
    patch_w_ = std::move(w0);
    if (!read_t(gg, "v.patch_embd.bias", patch_b_, err) || !read_t(gg, "v.position_embd.weight", pos_, err) ||
        !read_t(gg, "v.post_ln.weight", post_w_, err) || !read_t(gg, "v.post_ln.bias", post_b_, err) ||
        !read_t(gg, "mm.0.weight", mm0_w_, err) || !read_t(gg, "mm.0.bias", mm0_b_, err) ||
        !read_t(gg, "mm.2.weight", mm2_w_, err) || !read_t(gg, "mm.2.bias", mm2_b_, err))
        return false;
    act_round_ = 0;
    if (const char* e = std::getenv("STRATA_VIT_REF_ROUND"); e && std::atoi(e) != 0) {
        const GgufTensorInfo* t = gg.tensor("v.blk.0.attn_qkv.weight");
        act_round_ = t && t->type == GT_BF16 ? 1 : t && t->type == GT_F16 ? 2 : 0;
    }
    layers_.assign((size_t) g.layers, Layer{});
    for (int il = 0; il < g.layers; ++il) {
        Layer& L = layers_[(size_t) il];
        const std::string p = "v.blk." + std::to_string(il) + ".";
        if (!read_t(gg, p + "ln1.weight", L.ln1_w, err) || !read_t(gg, p + "ln1.bias", L.ln1_b, err) ||
            !read_t(gg, p + "ln2.weight", L.ln2_w, err) || !read_t(gg, p + "ln2.bias", L.ln2_b, err) ||
            !read_t(gg, p + "attn_qkv.weight", L.qkv_w, err) || !read_t(gg, p + "attn_qkv.bias", L.qkv_b, err) ||
            !read_t(gg, p + "attn_out.weight", L.o_w, err) || !read_t(gg, p + "attn_out.bias", L.o_b, err) ||
            !read_t(gg, p + "ffn_up.weight", L.up_w, err) || !read_t(gg, p + "ffn_up.bias", L.up_b, err) ||
            !read_t(gg, p + "ffn_down.weight", L.down_w, err) || !read_t(gg, p + "ffn_down.bias", L.down_b, err))
            return false;
    }
    return true;
}

bool VitRef::encode(const PreparedImage& img, int threads, std::vector<float>& rows, int& nx, int& ny, std::string& err) const {
    const VitGeometry& g = g_;
    const int P = g.patch, D = g.width, H = g.heads, HD = g.head_dim, F = g.ffn;
    if (layers_.empty()) { err = "the tower is not loaded"; return false; }
    if (HD > 128) { err = "head size above 128 is not handled by the reference"; return false; }
    if (img.w <= 0 || img.h <= 0 || img.w % g.align() != 0 || img.h % g.align() != 0) { err = "the picture size is not a multiple of patch * merge"; return false; }
    const int nth = threads > 0 ? threads : (int) std::max(1u, std::thread::hardware_concurrency());
    const int pw = img.w / P, ph = img.h / P, N = pw * ph, nbx = pw / 2;
    nx = pw / 2;
    ny = ph / 2;

    // ---- positions of the tokens (merge order: 2x2 blocks row by row, inside a block dy outer, dx inner) ----
    std::vector<int> ty((size_t) N), tx((size_t) N);
    for (int t = 0; t < N; ++t) {
        const int b = t / 4, r = t % 4;
        ty[(size_t) t] = 2 * (b / nbx) + r / 2;
        tx[(size_t) t] = 2 * (b % nbx) + r % 2;
    }

    // ---- patch embedding: im2col in merge order, one GEMM ----
    const int K0 = 3 * P * P;
    std::vector<float> A((size_t) N * K0);
    const size_t plane = (size_t) img.w * img.h;
    parallel_for(N, nth, [&](int t) {
        float* a = &A[(size_t) t * K0];
        const int py = ty[(size_t) t], px = tx[(size_t) t];
        for (int c = 0; c < 3; ++c)
            for (int ky = 0; ky < P; ++ky)
                std::memcpy(a + (size_t) c * P * P + (size_t) ky * P, &img.chw[c * plane + (size_t) (py * P + ky) * img.w + (size_t) px * P], (size_t) P * sizeof(float));
    });
    std::vector<float> x((size_t) N * D);
    gemm(A.data(), (size_t) K0, N, K0, patch_w_.data(), D, patch_b_.data(), x.data(), (size_t) D, nth);
    A.clear();
    A.shrink_to_fit();

    // ---- the position table resized to the patch grid (bilinear, align corners: ggml's upscale with the same float maths), added ----
    {
        const int G = g.pos_grid;
        float sfx = (float) pw / G, sfy = (float) ph / G;
        if (pw > 1 && G > 1) sfx = (float) (pw - 1) / (G - 1);
        if (ph > 1 && G > 1) sfy = (float) (ph - 1) / (G - 1);
        parallel_for(N, nth, [&](int t) {
            const float fy = (float) ty[(size_t) t] / sfy, fx = (float) tx[(size_t) t] / sfx;
            int y0 = (int) std::floor(fy), x0 = (int) std::floor(fx);
            int y1 = y0 + 1, x1 = x0 + 1;
            y0 = std::max(0, std::min(y0, G - 1)); y1 = std::max(0, std::min(y1, G - 1));
            x0 = std::max(0, std::min(x0, G - 1)); x1 = std::max(0, std::min(x1, G - 1));
            const float dy = std::max(0.0f, std::min(fy - (float) y0, 1.0f)), dx = std::max(0.0f, std::min(fx - (float) x0, 1.0f));
            const float* a = &pos_[((size_t) y0 * G + x0) * D];
            const float* b = &pos_[((size_t) y0 * G + x1) * D];
            const float* c = &pos_[((size_t) y1 * G + x0) * D];
            const float* d = &pos_[((size_t) y1 * G + x1) * D];
            float* xr = &x[(size_t) t * D];
            for (int i = 0; i < D; ++i) xr[i] += a[i] * (1 - dx) * (1 - dy) + b[i] * dx * (1 - dy) + c[i] * (1 - dx) * dy + d[i] * dx * dy;
        });
    }

    // ---- rotary tables: pair j of a head (elements j and j + HD/2), j < HD/4 turns with the row, the rest with the column ----
    const int half = HD / 2, quarter = HD / 4;
    std::vector<float> inv_freq((size_t) quarter);
    {
        const float theta_scale = std::pow(g.rope_theta, -2.0f / (float) half);
        float th = 1.0f;
        for (int j = 0; j < quarter; ++j) { inv_freq[(size_t) j] = th; th *= theta_scale; }
    }
    std::vector<float> rcos((size_t) N * half), rsin((size_t) N * half);
    parallel_for(N, nth, [&](int t) {
        for (int j = 0; j < half; ++j) {
            const float pos = (float) (j < quarter ? ty[(size_t) t] : tx[(size_t) t]);
            const float ang = pos * inv_freq[(size_t) (j % quarter)];
            rcos[(size_t) t * half + j] = std::cos(ang);
            rsin[(size_t) t * half + j] = std::sin(ang);
        }
    });

    // ---- the layers ----
    std::vector<float> h((size_t) N * D), qkv((size_t) N * 3 * D), ctx((size_t) N * D), tmp((size_t) N * D), up((size_t) N * F);
    const float kq_scale = 1.0f / std::sqrt((float) HD);
    for (const Layer& L : layers_) {
        layer_norm(x.data(), h.data(), N, D, L.ln1_w.data(), L.ln1_b.data(), g.eps, nth);
        round_inplace(h.data(), h.size(), act_round_, nth);
        gemm(h.data(), (size_t) D, N, D, L.qkv_w.data(), 3 * D, L.qkv_b.data(), qkv.data(), (size_t) 3 * D, nth);
        // rotary on q and k
        parallel_for(N, nth, [&](int t) {
            float* row = &qkv[(size_t) t * 3 * D];
            const float* cs = &rcos[(size_t) t * half];
            const float* sn = &rsin[(size_t) t * half];
            for (int part = 0; part < 2; ++part)
                for (int hh = 0; hh < H; ++hh) {
                    float* v = row + (size_t) part * D + (size_t) hh * HD;
                    for (int j = 0; j < half; ++j) {
                        const float a = v[j], b = v[j + half];
                        v[j] = a * cs[j] - b * sn[j];
                        v[j + half] = a * sn[j] + b * cs[j];
                    }
                }
        });
        round_inplace(qkv.data(), qkv.size(), act_round_ ? 2 : 0, nth);   // (diagnostic) mtmd's flash attention holds q, k, v in F16
        // global attention, 4 queries at a time per head
        const int QB = 4, nqb = (N + QB - 1) / QB;
        parallel_for(H * nqb, nth, [&](int task) {
            const int hh = task / nqb, q0 = (task % nqb) * QB, nq = std::min(QB, N - q0);
            std::vector<float> sc((size_t) QB * N);
            const size_t stride = (size_t) 3 * D;
            const float* qb = qkv.data() + (size_t) hh * HD;
            const float* kb = qkv.data() + (size_t) D + (size_t) hh * HD;
            const float* vb = qkv.data() + (size_t) 2 * D + (size_t) hh * HD;
            for (int j = 0; j < N; ++j) {
                const float* kj = kb + (size_t) j * stride;
                for (int qi = 0; qi < nq; ++qi) sc[(size_t) qi * N + j] = dot(qb + (size_t) (q0 + qi) * stride, kj, HD) * kq_scale;
            }
            for (int qi = 0; qi < nq; ++qi) {
                float* s = &sc[(size_t) qi * N];
                float mx = s[0];
                for (int j = 1; j < N; ++j) mx = std::max(mx, s[j]);
                double den = 0;
                for (int j = 0; j < N; ++j) { s[j] = std::exp(s[j] - mx); den += s[j]; }
                const float inv = (float) (1.0 / den);
                for (int j = 0; j < N; ++j) s[j] *= inv;
            }
            float acc[QB][128];
            for (int qi = 0; qi < nq; ++qi) for (int i = 0; i < HD; ++i) acc[qi][i] = 0.0f;
            for (int j = 0; j < N; ++j) {
                const float* vj = vb + (size_t) j * stride;
                for (int qi = 0; qi < nq; ++qi) {
                    const float p = sc[(size_t) qi * N + j];
                    for (int i = 0; i < HD; ++i) acc[qi][i] += p * vj[i];
                }
            }
            for (int qi = 0; qi < nq; ++qi) std::memcpy(&ctx[(size_t) (q0 + qi) * D + (size_t) hh * HD], acc[qi], (size_t) HD * sizeof(float));
        });
        round_inplace(ctx.data(), ctx.size(), act_round_, nth);
        gemm(ctx.data(), (size_t) D, N, D, L.o_w.data(), D, L.o_b.data(), tmp.data(), (size_t) D, nth);
        for (size_t i = 0; i < x.size(); ++i) x[i] += tmp[i];
        layer_norm(x.data(), h.data(), N, D, L.ln2_w.data(), L.ln2_b.data(), g.eps, nth);
        round_inplace(h.data(), h.size(), act_round_, nth);
        gemm(h.data(), (size_t) D, N, D, L.up_w.data(), F, L.up_b.data(), up.data(), (size_t) F, nth);
        gelu_inplace(up.data(), up.size(), nth);
        round_inplace(up.data(), up.size(), act_round_, nth);
        gemm(up.data(), (size_t) F, N, F, L.down_w.data(), D, L.down_b.data(), tmp.data(), (size_t) D, nth);
        for (size_t i = 0; i < x.size(); ++i) x[i] += tmp[i];
    }

    // ---- post norm and the merger ----
    layer_norm(x.data(), h.data(), N, D, post_w_.data(), post_b_.data(), g.eps, nth);
    const int M = N / 4, Kin = 4 * D, Hid = g.merger_hidden, O = g.out_dim;
    std::vector<float> m1((size_t) M * Hid);
    round_inplace(h.data(), h.size(), act_round_, nth);
    gemm(h.data(), (size_t) Kin, M, Kin, mm0_w_.data(), Hid, mm0_b_.data(), m1.data(), (size_t) Hid, nth);
    gelu_inplace(m1.data(), m1.size(), nth);
    rows.assign((size_t) M * O, 0.0f);
    round_inplace(m1.data(), m1.size(), act_round_, nth);
    gemm(m1.data(), (size_t) Hid, M, Hid, mm2_w_.data(), O, mm2_b_.data(), rows.data(), (size_t) O, nth);
    return true;
}

}  // namespace strata_vit
