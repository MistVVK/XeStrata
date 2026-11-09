// tools/vision/own/cpu_kernels_impl.h - the kernels of one ISA tier, written once over a small vector type V.  Included by cpu_kernels_<isa>.cpp after it has
// defined (before the include):
//   struct V        float vector of V::W lanes: load(p) (unaligned), store(p), set1(x), zero(), operators + - *, fma(a, b, c) = a*b + c, vmax(a, b), reduce_add(v),
//                   reduce_max(v), vexp(v) (expf, relative error below 3e-7 on [-87, 88])
//   void conv16(const void* W, int kind, size_t off, int n, float* out)     n elements from W[off] (kind 1 BF16, 2 F16) to float
//   STRATA_KN_MR    the rows of the register tile (NR is two vectors)
//   STRATA_KN_NAME  the tier's name, STRATA_KN_GETTER the table getter's name
#pragma once

#include "cpu_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace strata_vit {
namespace {      // every tier's kernels are internal to its translation unit (the vector types of the tiers share a name)
namespace kn {

constexpr int W = V::W;
constexpr int NR = 2 * W;
constexpr int MRF = STRATA_KN_MR;

// ---- the register-tiled product ----
template <int MR>
inline void micro(const float* A, size_t lda, const float* B, size_t ldb, int K, float* C, size_t ldc, int init, const float* bias) {
    V acc[MR][2];
    for (int r = 0; r < MR; ++r)
        for (int c = 0; c < 2; ++c) {
            V v = V::zero();
            if (init != 0) v = V::load(C + (size_t) r * ldc + c * W);
            if (init != 1 && bias) v = v + V::load(bias + c * W);
            acc[r][c] = v;
        }
    for (int k = 0; k < K; ++k) {
        const V b0 = V::load(B + (size_t) k * ldb), b1 = V::load(B + (size_t) k * ldb + W);
        for (int r = 0; r < MR; ++r) {
            const V a = V::set1(A[(size_t) r * lda + k]);
            acc[r][0] = V::fma(a, b0, acc[r][0]);
            acc[r][1] = V::fma(a, b1, acc[r][1]);
        }
    }
    for (int r = 0; r < MR; ++r)
        for (int c = 0; c < 2; ++c) acc[r][c].store(C + (size_t) r * ldc + c * W);
}

// m rows (m <= MRF) of one column tile
inline void tile_rows(const float* A, size_t lda, int m, const float* B, size_t ldb, int K, float* C, size_t ldc, int init, const float* bias) {
    int r = 0;
    for (; r + MRF <= m; r += MRF) micro<MRF>(A + (size_t) r * lda, lda, B, ldb, K, C + (size_t) r * ldc, ldc, init, bias);
    for (; r + 4 <= m; r += 4) micro<4>(A + (size_t) r * lda, lda, B, ldb, K, C + (size_t) r * ldc, ldc, init, bias);
    for (; r + 2 <= m; r += 2) micro<2>(A + (size_t) r * lda, lda, B, ldb, K, C + (size_t) r * ldc, ldc, init, bias);
    for (; r < m; ++r) micro<1>(A + (size_t) r * lda, lda, B, ldb, K, C + (size_t) r * ldc, ldc, init, bias);
}

void gemm(const float* A, size_t lda, int M, const float* B, size_t ldb, int K, int N, float* C, size_t ldc, int init, const float* bias) {
    for (int j0 = 0; j0 < N; j0 += NR) {
        const int nc = std::min(NR, N - j0);
        if (nc == NR) {
            tile_rows(A, lda, M, B + j0, ldb, K, C + j0, ldc, init, bias ? bias + j0 : nullptr);
        } else {      // the last, partial column tile: through a full-width buffer
            float bb[NR] = {0}, cb[MRF][NR];
            if (bias) std::memcpy(bb, bias + j0, (size_t) nc * sizeof(float));
            for (int m0 = 0; m0 < M; m0 += MRF) {
                const int mr = std::min(MRF, M - m0);
                for (int r = 0; r < mr; ++r) {
                    std::memset(cb[r], 0, sizeof cb[r]);
                    if (init != 0) std::memcpy(cb[r], C + (size_t) (m0 + r) * ldc + j0, (size_t) nc * sizeof(float));
                }
                tile_rows(A + (size_t) m0 * lda, lda, mr, B + j0, ldb, K, &cb[0][0], NR, init, bias ? bb : nullptr);
                for (int r = 0; r < mr; ++r) std::memcpy(C + (size_t) (m0 + r) * ldc + j0, cb[r], (size_t) nc * sizeof(float));
            }
        }
    }
}

void pack_w(const void* Wm, int kind, size_t ldw, int n0, int nvalid, int k0, int kc, float* dst) {
    float tmp[1024];
    for (int c = 0; c < NR; ++c) {
        if (c >= nvalid) {
            for (int k = 0; k < kc; ++k) dst[(size_t) k * NR + c] = 0.0f;
            continue;
        }
        const size_t off = (size_t) (n0 + c) * ldw + (size_t) k0;
        if (kind == 0) {
            const float* p = (const float*) Wm + off;
            for (int k = 0; k < kc; ++k) dst[(size_t) k * NR + c] = p[k];
        } else {
            for (int k = 0; k < kc; k += 1024) {
                const int n = std::min(1024, kc - k);
                conv16(Wm, kind, off + (size_t) k, n, tmp);
                for (int i = 0; i < n; ++i) dst[(size_t) (k + i) * NR + c] = tmp[i];
            }
        }
    }
}

void conv(const void* src, int kind, size_t n, float* dst) {
    for (size_t i = 0; i < n; i += 4096) conv16(src, kind, i, (int) std::min<size_t>(4096, n - i), dst + i);
}

// ---- elementwise ----
inline V gelu_v(V x) {
    // 0.5 x (1 + tanh(u)), u = sqrt(2/pi) x (1 + 0.044715 x^2);  tanh(u) = 1 - 2 / (exp(2u) + 1)
    const V one = V::set1(1.0f), two = V::set1(2.0f);
    const V u = V::set1(0.79788456080286535587989211986876f) * x * (one + V::set1(0.044715f) * x * x);
    const V e = vexp(two * u);
    return V::set1(0.5f) * x * (two - two / (e + one));      // 1 + tanh(u) = 2 - 2 / (exp(2u) + 1)
}

void gelu(float* x, size_t ldx, int rows, int n) {
    for (int r = 0; r < rows; ++r) {
        float* p = x + (size_t) r * ldx;
        int i = 0;
        for (; i + W <= n; i += W) gelu_v(V::load(p + i)).store(p + i);
        if (i < n) {
            float buf[W] = {0};
            std::memcpy(buf, p + i, (size_t) (n - i) * sizeof(float));
            gelu_v(V::load(buf)).store(buf);
            std::memcpy(p + i, buf, (size_t) (n - i) * sizeof(float));
        }
    }
}

void layernorm(const float* x, float* y, int D, const float* w, const float* b, float eps) {
    V s = V::zero();
    int i = 0;
    for (; i + W <= D; i += W) s = s + V::load(x + i);
    float sum = V::reduce_add(s);
    for (; i < D; ++i) sum += x[i];
    const float mean = sum / (float) D;
    const V vm = V::set1(mean);
    V q = V::zero();
    for (i = 0; i + W <= D; i += W) {
        const V d = V::load(x + i) - vm;
        q = V::fma(d, d, q);
    }
    float ss = V::reduce_add(q);
    for (; i < D; ++i) { const float d = x[i] - mean; ss += d * d; }
    const float scale = 1.0f / std::sqrt(ss / (float) D + eps);
    const V vs = V::set1(scale);
    for (i = 0; i + W <= D; i += W) (((V::load(x + i) - vm) * vs) * V::load(w + i) + V::load(b + i)).store(y + i);
    for (; i < D; ++i) y[i] = (x[i] - mean) * scale * w[i] + b[i];
}

void add(float* x, const float* y, size_t n) {
    size_t i = 0;
    for (; i + W <= n; i += W) (V::load(x + i) + V::load(y + i)).store(x + i);
    for (; i < n; ++i) x[i] += y[i];
}

void round16(float* p, size_t n, int kind) { round_buf(p, n, kind); }

void softmax_update(float* S, size_t lds, int qrows, int nkeys, float* m, float* l, float* O, size_t ldo, int hd) {
    for (int r = 0; r < qrows; ++r) {
        float* s = S + (size_t) r * lds;
        V mxv = V::set1(-3.0e38f);
        int j = 0;
        for (; j + W <= nkeys; j += W) mxv = vmax(mxv, V::load(s + j));
        float mx = V::reduce_max(mxv);
        for (; j < nkeys; ++j) mx = std::max(mx, s[j]);
        const float mold = m[r], mnew = std::max(mold, mx);
        const float alpha = mold < -1.0e38f ? 0.0f : std::exp(mold - mnew);
        float* o = O + (size_t) r * ldo;
        if (alpha != 1.0f) {
            const V va = V::set1(alpha);
            int d = 0;
            for (; d + W <= hd; d += W) (V::load(o + d) * va).store(o + d);
            for (; d < hd; ++d) o[d] *= alpha;
        }
        const V vmn = V::set1(mnew);
        V sv = V::zero();
        for (j = 0; j + W <= nkeys; j += W) {
            const V p = vexp(V::load(s + j) - vmn);
            p.store(s + j);
            sv = sv + p;
        }
        float sum = V::reduce_add(sv);
        for (; j < nkeys; ++j) { s[j] = std::exp(s[j] - mnew); sum += s[j]; }
        l[r] = l[r] * alpha + sum;
        m[r] = mnew;
    }
}

}  // namespace kn
}  // namespace

const CpuOps& STRATA_KN_GETTER() {
    static const CpuOps ops = {STRATA_KN_NAME, kn::MRF, kn::NR, kn::pack_w, kn::conv, kn::gemm, kn::gelu, kn::layernorm, kn::add, kn::round16, kn::softmax_update};
    return ops;
}

}  // namespace strata_vit
