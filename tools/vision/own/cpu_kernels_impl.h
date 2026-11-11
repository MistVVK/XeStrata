// tools/vision/own/cpu_kernels_impl.h - the kernels of one ISA tier, written once over a small vector type V.  Included by cpu_kernels_<isa>.cpp after it has
// defined (before the include):
//   struct V        float vector of V::W lanes: load(p) (unaligned), store(p), set1(x), zero(), operators + - *, fma(a, b, c) = a*b + c, vmax(a, b), reduce_add(v),
//                   reduce_max(v), vexp(v) (expf, relative error below 3e-7 on [-87, 88])
//   void conv16(const void* W, int kind, size_t off, int n, float* out)     n elements from W[off] (kind 1 BF16, 2 F16) to float
//   STRATA_KN_MR    the rows of the register tile (NR is two vectors)
// or, instead of struct V (STRATA_KN_RAW; the x86 tiers): the elementwise code below runs on a raw vector type VR with free inline functions VW (lanes),
//   vr_load, vr_store(p, v), vr_set1, vr_zero, vr_add, vr_sub, vr_mul, vr_div, vr_fma(a, b, c) = a*b + c, vr_max, vr_hadd, vr_hmax, vr_exp, vr_sqrt1, and the tier
//   brings its own micro (STRATA_KN_HAS_MICRO).  MSVC kept the wrapper struct in memory (RESULTS-OWN-3V.md 12.14); for the other tiers VR is V through the
//   adapters below.  Both forms are the same operations in the same order, so the results are bit-identical.
//   STRATA_KN_NAME  the tier's name, STRATA_KN_GETTER the table getter's name
//   KN_MIN(a, b), KN_MAX(a, b), KN_COPY(dst, src, n floats), KN_ZERO(dst, n floats), V::sqrt1(x) (a correctly rounded float square root)
// No standard-library header and no inline library function here (cpu_tier.h says why): the wider tiers define the KN_ helpers as their own internal
// functions; the scalar tier, compiled at the baseline, as std::min / std::max / std::memcpy / std::memset (the code GCC vectorises best there: with
// look-alike helpers the portable product loop came out partly scalar, 2 to 3 times slower, z440 11-10).
#pragma once

#include "cpu_tier.h"

namespace strata_vit {
namespace {      // every tier's kernels are internal to its translation unit (the vector types of the tiers share a name)
namespace kn {

#ifdef STRATA_KN_RAW
constexpr int W = VW;
#else
constexpr int W = V::W;
using VR = V;
inline VR vr_load(const float* p) { return V::load(p); }
inline void vr_store(float* p, VR v) { v.store(p); }
inline VR vr_set1(float f) { return V::set1(f); }
inline VR vr_zero() { return V::zero(); }
inline VR vr_add(VR a, VR b) { return a + b; }
inline VR vr_sub(VR a, VR b) { return a - b; }
inline VR vr_mul(VR a, VR b) { return a * b; }
inline VR vr_div(VR a, VR b) { return a / b; }
inline VR vr_fma(VR a, VR b, VR c) { return V::fma(a, b, c); }
inline VR vr_max(VR a, VR b) { return vmax(a, b); }
inline float vr_hadd(VR a) { return V::reduce_add(a); }
inline float vr_hmax(VR a) { return V::reduce_max(a); }
inline VR vr_exp(VR a) { return vexp(a); }
inline float vr_sqrt1(float x) { return V::sqrt1(x); }
#endif
constexpr int NR = 2 * W;
constexpr int MRF = STRATA_KN_MR;

// ---- the register-tiled product (a tier may bring its own `micro` with the same summation order: STRATA_KN_HAS_MICRO) ----
#ifndef STRATA_KN_HAS_MICRO
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
#endif

// m rows (m <= MRF) of one column tile
inline void tile_rows(const float* A, size_t lda, int m, const float* B, size_t ldb, int K, float* C, size_t ldc, int init, const float* bias) {
    int r = 0;
    for (; r + MRF <= m; r += MRF) micro<MRF>(A + (size_t) r * lda, lda, B, ldb, K, C + (size_t) r * ldc, ldc, init, bias);
    for (; r + 4 <= m; r += 4) micro<4>(A + (size_t) r * lda, lda, B, ldb, K, C + (size_t) r * ldc, ldc, init, bias);
    for (; r + 2 <= m; r += 2) micro<2>(A + (size_t) r * lda, lda, B, ldb, K, C + (size_t) r * ldc, ldc, init, bias);
    for (; r < m; ++r) micro<1>(A + (size_t) r * lda, lda, B, ldb, K, C + (size_t) r * ldc, ldc, init, bias);
}

#ifdef STRATA_KN_RAW
// The same product for a column tile of ONE vector (a last tile of at most W columns: the attention's P*V with a head size of 72 has 8 left over): each
// column's sum is the same FMA chain in the same order as in micro (one lane of the same intrinsics), so the results are bit-identical; half the work of a
// two-vector tile.  Only in the raw-vector tiers (AVX2, AVX-512), where vr_fma is the micro's own FMA intrinsic.
template <int MR>
inline void micro1(const float* A, size_t lda, const float* B, size_t ldb, int K, float* C, size_t ldc, int init, const float* bias) {
    VR acc[MR];
    for (int r = 0; r < MR; ++r) {
        VR v = vr_zero();
        if (init != 0) v = vr_load(C + (size_t) r * ldc);
        if (init != 1 && bias) v = vr_add(v, vr_load(bias));
        acc[r] = v;
    }
    for (int k = 0; k < K; ++k) {
        const VR b0 = vr_load(B + (size_t) k * ldb);
        for (int r = 0; r < MR; ++r) acc[r] = vr_fma(vr_set1(A[(size_t) r * lda + k]), b0, acc[r]);
    }
    for (int r = 0; r < MR; ++r) vr_store(C + (size_t) r * ldc, acc[r]);
}
inline void tile_rows1(const float* A, size_t lda, int m, const float* B, size_t ldb, int K, float* C, size_t ldc, int init, const float* bias) {
    int r = 0;
    for (; r + MRF <= m; r += MRF) micro1<MRF>(A + (size_t) r * lda, lda, B, ldb, K, C + (size_t) r * ldc, ldc, init, bias);
    for (; r + 4 <= m; r += 4) micro1<4>(A + (size_t) r * lda, lda, B, ldb, K, C + (size_t) r * ldc, ldc, init, bias);
    for (; r + 2 <= m; r += 2) micro1<2>(A + (size_t) r * lda, lda, B, ldb, K, C + (size_t) r * ldc, ldc, init, bias);
    for (; r < m; ++r) micro1<1>(A + (size_t) r * lda, lda, B, ldb, K, C + (size_t) r * ldc, ldc, init, bias);
}
#endif

void gemm(const float* A, size_t lda, int M, const float* B, size_t ldb, int K, int N, float* C, size_t ldc, int init, const float* bias) {
    for (int j0 = 0; j0 < N; j0 += NR) {
        const int nc = KN_MIN(NR, N - j0);
        if (nc == NR) {
            tile_rows(A, lda, M, B + j0, ldb, K, C + j0, ldc, init, bias ? bias + j0 : nullptr);
        }
#ifdef STRATA_KN_RAW
        else if (nc <= W && cpu_tier_one_vector_tail()) {      // a last tile of one vector or less: the one-vector product, through a one-vector buffer
            float bb[W] = {0}, cb[MRF][W];
            if (bias) KN_COPY(bb, bias + j0, nc);
            for (int m0 = 0; m0 < M; m0 += MRF) {
                const int mr = KN_MIN(MRF, M - m0);
                for (int r = 0; r < mr; ++r) {
                    KN_ZERO(cb[r], W);
                    if (init != 0) KN_COPY(cb[r], C + (size_t) (m0 + r) * ldc + j0, nc);
                }
                tile_rows1(A + (size_t) m0 * lda, lda, mr, B + j0, ldb, K, &cb[0][0], W, init, bias ? bb : nullptr);
                for (int r = 0; r < mr; ++r) KN_COPY(C + (size_t) (m0 + r) * ldc + j0, cb[r], nc);
            }
        }
#endif
        else {      // the last, partial column tile: through a full-width buffer
            float bb[NR] = {0}, cb[MRF][NR];
            if (bias) KN_COPY(bb, bias + j0, nc);
            for (int m0 = 0; m0 < M; m0 += MRF) {
                const int mr = KN_MIN(MRF, M - m0);
                for (int r = 0; r < mr; ++r) {
                    KN_ZERO(cb[r], NR);
                    if (init != 0) KN_COPY(cb[r], C + (size_t) (m0 + r) * ldc + j0, nc);
                }
                tile_rows(A + (size_t) m0 * lda, lda, mr, B + j0, ldb, K, &cb[0][0], NR, init, bias ? bb : nullptr);
                for (int r = 0; r < mr; ++r) KN_COPY(C + (size_t) (m0 + r) * ldc + j0, cb[r], nc);
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
                const int n = KN_MIN(1024, kc - k);
                conv16(Wm, kind, off + (size_t) k, n, tmp);
                for (int i = 0; i < n; ++i) dst[(size_t) (k + i) * NR + c] = tmp[i];
            }
        }
    }
}

void conv(const void* src, int kind, size_t n, float* dst) {
    for (size_t i = 0; i < n; i += 4096) conv16(src, kind, i, (int) KN_MIN((size_t) 4096, n - i), dst + i);
}

// ---- elementwise ----
inline VR gelu_v(VR x) {
    // 0.5 x (1 + tanh(u)), u = sqrt(2/pi) x (1 + 0.044715 x^2);  tanh(u) = 1 - 2 / (exp(2u) + 1)
    const VR one = vr_set1(1.0f), two = vr_set1(2.0f);
    const VR u = vr_mul(vr_mul(vr_set1(0.79788456080286535587989211986876f), x), vr_add(one, vr_mul(vr_mul(vr_set1(0.044715f), x), x)));
    const VR e = vr_exp(vr_mul(two, u));
    return vr_mul(vr_mul(vr_set1(0.5f), x), vr_sub(two, vr_div(two, vr_add(e, one))));      // 1 + tanh(u) = 2 - 2 / (exp(2u) + 1)
}

void gelu(float* x, size_t ldx, int rows, int n) {
    for (int r = 0; r < rows; ++r) {
        float* p = x + (size_t) r * ldx;
        int i = 0;
        for (; i + W <= n; i += W) vr_store(p + i, gelu_v(vr_load(p + i)));
        if (i < n) {
            float buf[W] = {0};
            KN_COPY(buf, p + i, n - i);
            vr_store(buf, gelu_v(vr_load(buf)));
            KN_COPY(p + i, buf, n - i);
        }
    }
}

void layernorm(const float* x, float* y, int D, const float* w, const float* b, float eps) {
    VR s = vr_zero();
    int i = 0;
    for (; i + W <= D; i += W) s = vr_add(s, vr_load(x + i));
    float sum = vr_hadd(s);
    for (; i < D; ++i) sum += x[i];
    const float mean = sum / (float) D;
    const VR vm = vr_set1(mean);
    VR q = vr_zero();
    for (i = 0; i + W <= D; i += W) {
        const VR d = vr_sub(vr_load(x + i), vm);
        q = vr_fma(d, d, q);
    }
    float ss = vr_hadd(q);
    for (; i < D; ++i) { const float d = x[i] - mean; ss += d * d; }
    const float scale = 1.0f / vr_sqrt1(ss / (float) D + eps);
    const VR vs = vr_set1(scale);
    for (i = 0; i + W <= D; i += W) vr_store(y + i, vr_add(vr_mul(vr_mul(vr_sub(vr_load(x + i), vm), vs), vr_load(w + i)), vr_load(b + i)));
    for (; i < D; ++i) y[i] = (x[i] - mean) * scale * w[i] + b[i];
}

void add(float* x, const float* y, size_t n) {
    size_t i = 0;
    for (; i + W <= n; i += W) vr_store(x + i, vr_add(vr_load(x + i), vr_load(y + i)));
    for (; i < n; ++i) x[i] += y[i];
}

void round16(float* p, size_t n, int kind) { round_buf(p, n, kind); }

void softmax_update(float* S, size_t lds, int qrows, int nkeys, float* m, float* l, float* O, size_t ldo, int hd) {
    for (int r = 0; r < qrows; ++r) {
        float* s = S + (size_t) r * lds;
        VR mxv = vr_set1(-3.0e38f);
        int j = 0;
        for (; j + W <= nkeys; j += W) mxv = vr_max(mxv, vr_load(s + j));
        float mx = vr_hmax(mxv);
        for (; j < nkeys; ++j) mx = KN_MAX(mx, s[j]);
        const float mold = m[r], mnew = KN_MAX(mold, mx);
        const float alpha = mold < -1.0e38f ? 0.0f : cpu_tier_expf(mold - mnew);
        float* o = O + (size_t) r * ldo;
        if (alpha != 1.0f) {
            const VR va = vr_set1(alpha);
            int d = 0;
            for (; d + W <= hd; d += W) vr_store(o + d, vr_mul(vr_load(o + d), va));
            for (; d < hd; ++d) o[d] *= alpha;
        }
        const VR vmn = vr_set1(mnew);
        VR sv = vr_zero();
        for (j = 0; j + W <= nkeys; j += W) {
            const VR p = vr_exp(vr_sub(vr_load(s + j), vmn));
            vr_store(s + j, p);
            sv = vr_add(sv, p);
        }
        float sum = vr_hadd(sv);
        for (; j < nkeys; ++j) { s[j] = cpu_tier_expf(s[j] - mnew); sum += s[j]; }
        l[r] = l[r] * alpha + sum;
        m[r] = mnew;
    }
}

// The rotary and the position add, written as they were in vit_cpu.cpp (the same expressions in the same order), so a tier compiled with the flags vit_cpu.cpp
// had (-mavx2 -mfma on GCC) contracts them the same way as before: the AVX2 / AVX-512 results stay bit-identical.
void rope_row(float* row, size_t kstride, int heads, int hd, const float* c1, const float* s1, const float* c2, const float* s2) {
    const int half = hd / 2, quarter = hd / 4;
    for (int part = 0; part < 2; ++part)
        for (int hh = 0; hh < heads; ++hh) {
            float* v = row + (size_t) part * kstride + (size_t) hh * hd;
            for (int j = 0; j < half; ++j) {
                const float co = j < quarter ? c1[j] : c2[j - quarter], si = j < quarter ? s1[j] : s2[j - quarter];
                const float a = v[j], b = v[j + half];
                v[j] = a * co - b * si;
                v[j + half] = a * si + b * co;
            }
        }
}

void pos_add(float* xr, const float* a, const float* b, const float* c, const float* d, float dx, float dy, int D) {
    for (int i = 0; i < D; ++i) xr[i] += a[i] * (1 - dx) * (1 - dy) + b[i] * dx * (1 - dy) + c[i] * (1 - dx) * dy + d[i] * dx * dy;
}

}  // namespace kn
}  // namespace

const CpuOps& STRATA_KN_GETTER() {
    static const CpuOps ops = {STRATA_KN_NAME, kn::MRF, kn::NR, kn::pack_w, kn::conv, kn::gemm, kn::gelu, kn::layernorm, kn::add, kn::round16, kn::softmax_update,
                               kn::rope_row, kn::pos_add};
    return ops;
}

}  // namespace strata_vit
