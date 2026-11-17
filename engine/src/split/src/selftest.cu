// HIP only: the kernel self-test the engine runs on a card Strata has not run it on (hip/selftest_policy.hpp has the rule and the lines). Random small data, a scalar
// reference on the host, a few milliseconds each:
//   wmma     the nvcuda::wmma stand-in (hip/mma.h) in the four operand layouts the attention tiles use (Q K^T, P V, and the transposed forms), both stores
//   expert   the prompt path's routed-expert tile (moe_mma.cuh, Q8_0 weights) through the engine's own launcher, jobs of 40 and 5 tokens, 96 rows (a partial tile)
//   dense    the prompt path's dense product (hipBLAS f16 GEMM, f32 accumulation)
//   dp4a     the decode kernels' four-way int8 dot product (hip_compat's __dp4a)
//   gdn      the recurrent layer's step (gdn_step, the delta rule on a 128 x 128 state per value head, wave shuffles)
// The inputs are small multiples of powers of two, so the f16 products are exact; the tensor cores add them in their own order and precision (gfx11's WMMA is a few 1e-7 off an exact
// f32 sum, measured on a 780M), so a value passes within 1e-4 of the sum of the absolute terms. A wrong operand or accumulator layout is off by whole products.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "kernels.hpp"
#include "model.hpp"
#include "selftest.hpp"

namespace fnx {
namespace {

struct Rng {
    unsigned long long s;
    explicit Rng(unsigned long long seed) : s(seed * 2685821657736338717ull + 1442695040888963407ull) {}
    unsigned next() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return (unsigned) ((s * 2685821657736338717ull) >> 32);
    }
    int range(int lo, int hi) { return lo + (int) (next() % (unsigned) (hi - lo + 1)); }   // inclusive
    float unit() { return (float) (next() & 0xffff) / 65536.0f; }
};

std::string cuerr(cudaError_t e, const char* what) { return std::string(what) + ": " + cudaGetErrorString(e); }

template <typename T> struct Dev {
    T* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t count) : n(count) {
        if (cudaMalloc((void**) &p, count * sizeof(T)) != cudaSuccess) p = nullptr, cudaGetLastError();
    }
    ~Dev() {
        if (p) cudaFree(p);
    }
    bool ok() const { return p != nullptr; }
    void up(const std::vector<T>& h) { cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice); }
    std::vector<T> down() const {
        std::vector<T> h(n);
        cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
        return h;
    }
};

uint16_t h16(float x) { return __half_as_ushort(__float2half(x)); }

// ---- wmma: C = A B for one 16 x 16 x 16 tile, A and B in the layout LA / LB (ldm 24), stored row- or column-major
constexpr int T1_LD = 24;
template <typename LA, typename LB>
__global__ void k_t1(const __half* a, const __half* b, float* c_row, float* c_col) {
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, LA> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, LB> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
    wmma::fill_fragment(fc, 0.f);
    wmma::load_matrix_sync(fa, a, T1_LD);
    wmma::load_matrix_sync(fb, b, T1_LD);
    wmma::mma_sync(fc, fa, fb, fc);
    wmma::store_matrix_sync(c_row, fc, 16, wmma::mem_row_major);
    wmma::store_matrix_sync(c_col, fc, 16, wmma::mem_col_major);
}
template <typename LA, typename LB> std::string t1_case(const char* name, bool a_row, bool b_col, const std::vector<float>& A, const std::vector<float>& B,
                                                         const __half* da, const __half* db, Dev<float>& cr, Dev<float>& cc) {
    cudaMemset(cr.p, 0, cr.n * 4);
    cudaMemset(cc.p, 0, cc.n * 4);
    k_t1<LA, LB><<<1, 32>>>(da, db, cr.p, cc.p);
    if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) return cuerr(e, name);
    const std::vector<float> r = cr.down(), c = cc.down();
    int bad = 0;
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 16; ++n) {
            float ref = 0.f;
            for (int k = 0; k < 16; ++k) {
                const float x = a_row ? A[m * T1_LD + k] : A[k * T1_LD + m];   // A[m][k]
                const float y = b_col ? B[n * T1_LD + k] : B[k * T1_LD + n];   // B[k][n]
                ref += x * y;
            }
            float mag = 1e-3f;
            for (int k = 0; k < 16; ++k) mag += std::fabs((a_row ? A[m * T1_LD + k] : A[k * T1_LD + m]) * (b_col ? B[n * T1_LD + k] : B[k * T1_LD + n]));
            if (std::fabs(r[m * 16 + n] - ref) > 1e-4f * mag) ++bad;
            if (std::fabs(c[n * 16 + m] - ref) > 1e-4f * mag) ++bad;
        }
    return bad ? std::string("wmma ") + name + ": " + std::to_string(bad) + " of 512 values wrong" : "";
}
std::string test_wmma() {
    Rng rng(11);
    std::vector<float> A(16 * T1_LD), B(16 * T1_LD);
    std::vector<uint16_t> ha(A.size()), hb(B.size());
    for (size_t i = 0; i < A.size(); ++i) {
        A[i] = (float) rng.range(-8, 8) / 4.f, B[i] = (float) rng.range(-8, 8) / 4.f;
        ha[i] = h16(A[i]), hb[i] = h16(B[i]);
    }
    Dev<uint16_t> da(A.size()), db(B.size());
    Dev<float> cr(256), cc(256);
    if (!da.ok() || !db.ok() || !cr.ok() || !cc.ok()) return "wmma: allocation failed";
    da.up(ha);
    db.up(hb);
    using nvcuda::wmma::row_major;
    using nvcuda::wmma::col_major;
    const __half* pa = reinterpret_cast<const __half*>(da.p);
    const __half* pb = reinterpret_cast<const __half*>(db.p);
    std::string e;
    if (!(e = t1_case<row_major, col_major>("A row, B col (Q K^T)", true, true, A, B, pa, pb, cr, cc)).empty()) return e;
    if (!(e = t1_case<row_major, row_major>("A row, B row (P V)", true, false, A, B, pa, pb, cr, cc)).empty()) return e;
    if (!(e = t1_case<col_major, col_major>("A col, B col", false, true, A, B, pa, pb, cr, cc)).empty()) return e;
    if (!(e = t1_case<col_major, row_major>("A col, B row", false, false, A, B, pa, pb, cr, cc)).empty()) return e;
    return "";
}

// ---- expert: moe_gate_up_mma on a Q8_0 matrix
std::string test_expert() {
    const int nrows = 96, ncols = 128, ntok = 45;
    Rng rng(22);
    const size_t row_bytes = (size_t) ncols / 32 * 34;
    std::vector<uint8_t> w((size_t) nrows * row_bytes);
    std::vector<float> wf((size_t) nrows * ncols);
    const float scales[3] = {0.25f, 0.125f, 0.0625f};
    for (int r = 0; r < nrows; ++r)
        for (int blk = 0; blk < ncols / 32; ++blk) {
            const float d = scales[rng.range(0, 2)];
            uint8_t* b = w.data() + (size_t) r * row_bytes + (size_t) blk * 34;
            const uint16_t dh = h16(d);
            std::memcpy(b, &dh, 2);
            for (int i = 0; i < 32; ++i) {
                const int q = rng.range(-127, 127);
                b[2 + i] = (uint8_t) (int8_t) q;
                wf[(size_t) r * ncols + blk * 32 + i] = d * (float) q;
            }
        }
    std::vector<uint16_t> xh((size_t) ntok * ncols);
    std::vector<float> xf(xh.size());
    for (size_t i = 0; i < xh.size(); ++i) {
        const float v = (float) rng.range(-32, 32) / 16.f;
        xf[i] = v, xh[i] = h16(v);
    }
    std::vector<BigJob> jobs(2);
    std::memset(jobs.data(), 0, jobs.size() * sizeof(BigJob));
    jobs[0].w = 0, jobs[0].n = 40;
    jobs[1].w = 0, jobs[1].n = 5;
    for (int j = 0; j < 40; ++j) jobs[0].y[j] = j, jobs[0].out[j] = j;
    for (int j = 0; j < 5; ++j) jobs[1].y[j] = 40 + j, jobs[1].out[j] = 40 + j;
    Dev<uint8_t> dw(w.size());
    Dev<uint16_t> dx(xh.size());
    Dev<BigJob> dj(2);
    Dev<int> dn(1);
    Dev<float> dout((size_t) ntok * nrows);
    if (!dw.ok() || !dx.ok() || !dj.ok() || !dn.ok() || !dout.ok()) return "expert tile: allocation failed";
    dw.up(w);
    dx.up(xh);
    dj.up(jobs);
    dn.up(std::vector<int>{2});
    cudaMemset(dout.p, 0, dout.n * 4);
    DW A;
    A.p = dw.p;
    A.type = T_Q8_0;
    A.n_in = ncols;
    A.n_out = nrows;
    A.row_bytes = row_bytes;
    try {
        moe_gate_up_mma(A, w.size(), dj.p, dn.p, 2, dx.p, dout.p, nullptr);
    } catch (const std::exception& ex) {
        return std::string("expert tile: ") + ex.what();
    }
    if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) return cuerr(e, "expert tile");
    const std::vector<float> out = dout.down();
    int bad = 0;
    for (int t = 0; t < ntok; ++t)
        for (int r = 0; r < nrows; ++r) {
            double ref = 0, mag = 1e-3;
            for (int k = 0; k < ncols; ++k) {
                const double term = (double) wf[(size_t) r * ncols + k] * (double) xf[(size_t) t * ncols + k];
                ref += term, mag += std::fabs(term);
            }
            if (std::fabs((double) out[(size_t) t * nrows + r] - ref) > 1e-4 * mag) ++bad;
        }
    return bad ? "expert tile: " + std::to_string(bad) + " of " + std::to_string(ntok * nrows) + " values wrong" : "";
}

// ---- dense: the prompt path's f16 GEMM (weights [rows][cols], activations [C][cols], f32 out [C][rows])
std::string test_dense() {
    const int rows = 80, cols = 96, C = 24;
    Rng rng(33);
    std::vector<uint16_t> w((size_t) rows * cols), x((size_t) C * cols);
    std::vector<float> wf(w.size()), xf(x.size());
    for (size_t i = 0; i < w.size(); ++i) wf[i] = (float) rng.range(-16, 16) / 8.f, w[i] = h16(wf[i]);
    for (size_t i = 0; i < x.size(); ++i) xf[i] = (float) rng.range(-16, 16) / 8.f, x[i] = h16(xf[i]);
    Dev<uint16_t> dw(w.size()), dx(x.size());
    Dev<float> dout((size_t) C * rows);
    if (!dw.ok() || !dx.ok() || !dout.ok()) return "dense product: allocation failed";
    dw.up(w);
    dx.up(x);
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) return "dense product: hipBLAS would not start";
    const float one = 1.f, zero = 0.f;
    const cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, rows, C, cols, &one, dw.p, CUDA_R_16F, cols, dx.p, CUDA_R_16F, cols, &zero, dout.p, CUDA_R_32F, rows,
                                           CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasDestroy(h);
    if (st != CUBLAS_STATUS_SUCCESS) return "dense product: hipBLAS GemmEx failed (status " + std::to_string((int) st) + ")";
    if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) return cuerr(e, "dense product");
    const std::vector<float> out = dout.down();
    int bad = 0;
    for (int t = 0; t < C; ++t)
        for (int r = 0; r < rows; ++r) {
            double ref = 0, mag = 1e-3;
            for (int k = 0; k < cols; ++k) {
                const double term = (double) wf[(size_t) r * cols + k] * (double) xf[(size_t) t * cols + k];
                ref += term, mag += std::fabs(term);
            }
            if (std::fabs((double) out[(size_t) t * rows + r] - ref) > 1e-4 * mag) ++bad;
        }
    return bad ? "dense product: " + std::to_string(bad) + " of " + std::to_string(C * rows) + " values wrong" : "";
}

// ---- dp4a
__global__ void k_t4(const int* a, const int* b, const int* c, int* out, int n) {
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += gridDim.x * blockDim.x) out[e] = __dp4a(a[e], b[e], c[e]);
}
std::string test_dp4a() {
    const int n = 4096;
    Rng rng(44);
    std::vector<int> a(n), b(n), c(n), ref(n);
    for (int i = 0; i < n; ++i) {
        a[i] = (int) rng.next(), b[i] = (int) rng.next(), c[i] = (int) rng.next();
        int s = c[i];
        for (int k = 0; k < 4; ++k) s += (int) (int8_t) ((a[i] >> (8 * k)) & 255) * (int) (int8_t) ((b[i] >> (8 * k)) & 255);
        ref[i] = s;
    }
    Dev<int> da(n), db(n), dc(n), dout(n);
    if (!da.ok() || !db.ok() || !dc.ok() || !dout.ok()) return "dp4a: allocation failed";
    da.up(a), db.up(b), dc.up(c);
    k_t4<<<16, 256>>>(da.p, db.p, dc.p, dout.p, n);
    if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) return cuerr(e, "dp4a");
    const std::vector<int> out = dout.down();
    int bad = 0;
    for (int i = 0; i < n; ++i) bad += out[i] != ref[i];
    return bad ? "dp4a: " + std::to_string(bad) + " of " + std::to_string(n) + " wrong" : "";
}

// ---- gdn: two steps of the delta rule against a double-precision reference
std::string test_gdn() {
    const int nk = NKH, nv = NVH, T = 2, ch = 2 * nk * HS + nv * HS;
    Rng rng(55);
    std::vector<float> S((size_t) nv * HS * HS), conv((size_t) T * ch), g((size_t) T * nv), beta((size_t) T * nv);
    for (auto& v : S) v = (rng.unit() - 0.5f) * 0.5f;
    for (auto& v : conv) v = (rng.unit() - 0.5f) * 0.5f;
    for (auto& v : g) v = -rng.unit();                 // exp(g) in (0.37, 1]
    for (auto& v : beta) v = 0.25f + rng.unit() * 0.5f;
    Dev<float> dS(S.size()), dc(conv.size()), dg(g.size()), db(beta.size()), dout((size_t) T * nv * HS);
    if (!dS.ok() || !dc.ok() || !dg.ok() || !db.ok() || !dout.ok()) return "recurrent step: allocation failed";
    dS.up(S), dc.up(conv), dg.up(g), db.up(beta);
    try {
        gdn_step(dS.p, dc.p, dg.p, db.p, dout.p, nk, nv, ch, T, nullptr);
    } catch (const std::exception& ex) {
        return std::string("recurrent step: ") + ex.what();
    }
    if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) return cuerr(e, "recurrent step");
    const std::vector<float> o = dout.down(), S1 = dS.down();
    // the reference: S[lv][col][r] (the kernel's layout: row `col` of the head's state, 128 floats), kv = S . k, delta = (v - g kv) beta, S = g S + k delta, o = S . q / sqrt(HS)
    const double scale = 1.0 / std::sqrt((double) HS);
    int bad = 0;
    std::vector<double> st(S.begin(), S.end());
    for (int lv = 0; lv < nv; lv += 7) {   // a spread of heads (all of them is 48 x 2 x 128 x 128 x 2 flops: cheap enough, but a few say it)
        const int kh = lv % nk;
        std::vector<double> s(st.begin() + (size_t) lv * HS * HS, st.begin() + (size_t) (lv + 1) * HS * HS);
        for (int t = 0; t < T; ++t) {
            const float* row = conv.data() + (size_t) t * ch;
            const double gv = std::exp((double) g[(size_t) t * nv + lv]), bv = beta[(size_t) t * nv + lv];
            for (int col = 0; col < HS; ++col) {
                double kv = 0;
                for (int r = 0; r < HS; ++r) kv += s[(size_t) col * HS + r] * row[nk * HS + kh * HS + r];
                const double delta = (row[2 * nk * HS + lv * HS + col] - gv * kv) * bv;
                double at = 0;
                for (int r = 0; r < HS; ++r) {
                    s[(size_t) col * HS + r] = gv * s[(size_t) col * HS + r] + row[nk * HS + kh * HS + r] * delta;
                    at += s[(size_t) col * HS + r] * row[kh * HS + r];
                }
                const double want = at * scale, got = o[(size_t) t * nv * HS + lv * HS + col];
                if (std::fabs(got - want) > 2e-3 * (1.0 + std::fabs(want))) ++bad;
            }
        }
        for (size_t i = 0; i < s.size(); i += 97)
            if (std::fabs((double) S1[(size_t) lv * HS * HS + i] - s[i]) > 2e-3 * (1.0 + std::fabs(s[i]))) ++bad;
    }
    return bad ? "recurrent step: " + std::to_string(bad) + " values wrong" : "";
}

}  // namespace

// the tests on device `dev`; "" when all passed (covered: what ran), else the first failure in a few words
std::string selftest_card(int dev, std::string& covered) {
    if (cudaError_t e = cudaSetDevice(dev); e != cudaSuccess) return cuerr(e, "cudaSetDevice");
    struct T {
        const char* name;
        std::string (*fn)();
    } const tests[] = {{"wmma", test_wmma}, {"expert tile", test_expert}, {"dense product", test_dense}, {"dp4a", test_dp4a}, {"recurrent step", test_gdn}};
    covered.clear();
    for (const T& t : tests) {
        const std::string bad = t.fn();
        if (!bad.empty()) return bad;
        cudaGetLastError();
        covered += (covered.empty() ? "" : ", ") + std::string(t.name);
    }
    covered = "card " + std::to_string(dev) + ": " + covered + " against a scalar reference";
    return "";
}

}  // namespace fnx
