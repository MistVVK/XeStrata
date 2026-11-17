// src/prefill/pf_cover_test.cu - FULL-OUTPUT coverage check of the gfx11 prompt GEMMs (wmma_gemm.cu): every output element is
// poisoned with NaN before the launch and must be written (finite) after it, and a stride-sampled set of rows (every
// 37th row, the first, the last 130, all columns) is compared with an FP64 host reference.  It guards against the
// failure class halo-box/strix-llama.cpp #213 fixed in its own WMMA kernels: a tile variant that covers only part of its
// output (a skipped second 16-token fragment, a 128-column tile that stops at 64) - which a few-row parity sample at a
// shape that picks another variant does not see.  Covered:
//   strata_wmma_gemm_f16 / _bf16       the 16x16 (1 wave) and 64x64 (4 waves) kernels       (T, N, K from tails to 8191)
//   strata_pf_gemm_f16_ld              kernel_bk64 (N >= 1024), kernel<256,64>, kernel<128,32> (N < 1024), padded strides
//   strata_pf_hcdown_bf16              the HC down + inject GEMM (nd 320 + ni 4, K 10240)
//   strata_pf_hcdown_exact_bf16        the hipBLASLt-order HC down GEMM
// Exit 77 without a gfx11 GPU.   pf_cover_test           the checks (a few minutes of GPU time at most)
#include "wmma_gemm.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

bool ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) std::printf("FAIL: %s: %s\n", what, cudaGetErrorString(e));
    return e == cudaSuccess;
}
uint16_t f16_bits(float f) {
    _Float16 h = (_Float16) f;
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}
float f16_val(uint16_t u) {
    _Float16 h;
    std::memcpy(&h, &u, 2);
    return (float) h;
}
uint16_t bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
float bf16_val(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct Mat {
    std::vector<uint16_t> bits;   // rows x ld
    std::vector<float> val;       // rows x ld (the value the bits stand for)
    uint16_t* dev = nullptr;
    int64_t rows = 0, ld = 0;
};
Mat make(int64_t rows, int64_t cols, int64_t ld, bool bf16, std::mt19937& rng, float scale) {
    Mat m;
    m.rows = rows;
    m.ld = ld;
    m.bits.assign((size_t) rows * ld, 0);
    m.val.assign((size_t) rows * ld, 0.0f);
    std::normal_distribution<float> nd(0.0f, scale);
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < cols; ++c) {
            const float x = nd(rng);
            const uint16_t b = bf16 ? bf16_bits(x) : f16_bits(x);
            m.bits[(size_t) r * ld + c] = b;
            m.val[(size_t) r * ld + c] = bf16 ? bf16_val(b) : f16_val(b);
        }
    if (!ck(cudaMalloc(&m.dev, m.bits.size() * 2), "malloc")) std::exit(2);
    ck(cudaMemcpy(m.dev, m.bits.data(), m.bits.size() * 2, cudaMemcpyHostToDevice), "up");
    return m;
}

struct Result {
    int64_t unwritten = 0, bad = 0, checked = 0;
    double worst = 0;
};

// Y [T x ldy], columns [c0, c0 + n) of row t from X rows and W rows
Result check_out(const std::vector<float>& y, int64_t T, int64_t ldy, const Mat& X, const Mat& W, int64_t w0, int64_t n,
                 int64_t K, int64_t c0) {
    Result r;
    for (int64_t t = 0; t < T; ++t)
        for (int64_t j = 0; j < n; ++j) {
            const float v = y[(size_t) t * ldy + c0 + j];
            if (!std::isfinite(v)) ++r.unwritten;
        }
    auto row_ok = [&](int64_t t) {
        double scale = 0;
        std::vector<double> ref((size_t) n);
        for (int64_t j = 0; j < n; ++j) {
            double s = 0, a = 0;
            for (int64_t k = 0; k < K; ++k) {
                const double p = (double) X.val[(size_t) t * X.ld + k] * (double) W.val[(size_t) (w0 + j) * W.ld + k];
                s += p;
                a += std::fabs(p);
            }
            ref[(size_t) j] = s;
            scale = std::max(scale, a);
        }
        for (int64_t j = 0; j < n; ++j) {
            const double d = std::fabs((double) y[(size_t) t * ldy + c0 + j] - ref[(size_t) j]) / (scale + 1e-30);
            r.worst = std::max(r.worst, d);
            if (!(d < 2e-4)) ++r.bad;   // FP32 accumulate over K: ~1e-6 of the absolute sum, 2e-4 leaves room for any order
            ++r.checked;
        }
    };
    for (int64_t t = 0; t < T; t += 37) row_ok(t);
    for (int64_t t = std::max<int64_t>(0, T - 130); t < T; ++t) row_ok(t);
    return r;
}

int fails = 0;
void report(const char* what, int64_t T, int64_t N, int64_t K, const Result& r) {
    const bool ok = r.unwritten == 0 && r.bad == 0;
    if (!ok) ++fails;
    std::printf("%-26s T %5lld N %5lld K %5lld  unwritten %lld  bad %lld of %lld  worst rel %.2e  %s\n", what, (long long) T,
                (long long) N, (long long) K, (long long) r.unwritten, (long long) r.bad, (long long) r.checked, r.worst,
                ok ? "OK" : "FAIL");
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    std::mt19937 rng(777);
    float* dy = nullptr;
    const size_t YMAX = (size_t) 8191 * 4128;   // T x ldy at the largest shapes
    if (!ck(cudaMalloc(&dy, YMAX * 4), "malloc y")) return 2;
    std::vector<float> y;
    auto fetch = [&](int64_t T, int64_t ldy) {
        y.resize((size_t) T * ldy);
        ck(cudaDeviceSynchronize(), "sync");
        ck(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost), "down");
    };
    bool any_ran = false;

    // ---- the generic WMMA GEMMs (T x K) . (N x K)^T, FP16 and BF16
    struct S { int64_t T, N, K; };
    const S generic[] = {{1, 16, 32}, {17, 40, 64}, {31, 31 + 1, 48}, {32, 32, 16}, {33, 96, 128}, {100, 130, 256},
                         {513, 200, 192}, {1000, 576, 512}, {4093, 2560, 256}, {8191, 512, 128}};
    for (bool bf : {false, true}) {
        for (const S& s : generic) {
            Mat X = make(s.T, s.K, s.K, bf, rng, 1.0f), W = make(s.N, s.K, s.K, bf, rng, 1.0f);
            ck(cudaMemset(dy, 0xff, (size_t) s.T * s.N * 4), "poison");
            const bool ran = bf ? strata_wmma_gemm_bf16(X.dev, W.dev, dy, s.T, s.N, s.K, s.N, 0.0f, nullptr)
                                : strata_wmma_gemm_f16(X.dev, W.dev, dy, s.T, s.N, s.K, s.N, 0.0f, nullptr);
            if (!ran) { std::printf("%s T %lld N %lld K %lld: not run (no gfx11 device?)\n", bf ? "wmma bf16" : "wmma f16", (long long) s.T, (long long) s.N, (long long) s.K); continue; }
            any_ran = true;
            fetch(s.T, s.N);
            report(bf ? "wmma_gemm_bf16" : "wmma_gemm_f16", s.T, s.N, s.K, check_out(y, s.T, s.N, X, W, 0, s.N, s.K, 0));
            cudaFree(X.dev);
            cudaFree(W.dev);
        }
    }

    // ---- the S23 128-row GEMM: bk64 (N >= 1024, K % 64), <256,64> (N >= 1024, K % 32 only), <128,32> (N < 1024); padded strides
    const S pf[] = {{64, 512, 64}, {100, 640, 128}, {513, 1000 - 1000 + 768, 192}, {1000, 1024, 256}, {4093, 1280, 96},
                    {4093, 2560, 2048}, {8191, 2560, 2048}, {8191, 1024, 4096}, {8191, 768, 2560}, {8191, 4096, 32 * 5}};
    for (const S& s : pf) {
        for (int pad : {0, 64}) {
            const int64_t ldx = s.K + pad, ldw = s.K + pad;
            Mat X = make(s.T, s.K, ldx, false, rng, 1.0f), W = make(s.N, s.K, ldw, false, rng, 1.0f);
            for (float beta : {0.0f, 1.0f}) {
                // beta 1: Y starts as a known finite value 1.0 so only an unwritten tile (left at 1.0 + nothing) differs
                if (beta == 0.0f) ck(cudaMemset(dy, 0xff, (size_t) s.T * s.N * 4), "poison");
                else {
                    std::vector<float> ones((size_t) s.T * s.N, 1.0f);
                    ck(cudaMemcpy(dy, ones.data(), ones.size() * 4, cudaMemcpyHostToDevice), "ones");
                }
                if (!strata_pf_gemm_f16_ld(X.dev, ldx, W.dev, ldw, dy, s.T, s.N, s.K, s.N, beta, nullptr)) {
                    std::printf("pf_gemm_f16_ld T %lld N %lld K %lld pad %d: not run\n", (long long) s.T, (long long) s.N, (long long) s.K, pad);
                    continue;
                }
                any_ran = true;
                fetch(s.T, s.N);
                if (beta == 1.0f) for (float& v : y) v -= 1.0f;   // the reference has no addend; an unwritten element stays 0.0 and fails the compare
                char name[64];
                std::snprintf(name, sizeof name, "pf_gemm pad %d beta %d", pad, (int) beta);
                report(name, s.T, s.N, s.K, check_out(y, s.T, s.N, X, W, 0, s.N, s.K, 0));
            }
            cudaFree(X.dev);
            cudaFree(W.dev);
        }
    }

    // ---- HC down + inject (nd 320 + ni 4, K 10240) and the hipBLASLt-order down (N 320): BF16, token stride 10240 + 64
    const int64_t HK = 10240, HLD = HK + 64;
    for (int64_t T : {64, 100, 513, 1000, 4093, 8191}) {
        Mat X = make(T, HK, HLD, true, rng, 1.0f), Wd = make(320, HK, HK, true, rng, 0.05f), Wi = make(4, HK, HK, true, rng, 0.05f);
        float *dlo = nullptr, *dinj = nullptr;
        ck(cudaMalloc(&dlo, (size_t) T * 320 * 4), "malloc");
        ck(cudaMalloc(&dinj, (size_t) T * 4 * 4), "malloc");
        ck(cudaMemset(dlo, 0xff, (size_t) T * 320 * 4), "poison");
        ck(cudaMemset(dinj, 0xff, (size_t) T * 4 * 4), "poison");
        if (strata_pf_hcdown_bf16(X.dev, HLD, Wd.dev, Wi.dev, 320, 4, dlo, dinj, T, HK, nullptr)) {
            any_ran = true;
            ck(cudaDeviceSynchronize(), "sync");
            std::vector<float> lo((size_t) T * 320), inj((size_t) T * 4);
            ck(cudaMemcpy(lo.data(), dlo, lo.size() * 4, cudaMemcpyDeviceToHost), "down");
            ck(cudaMemcpy(inj.data(), dinj, inj.size() * 4, cudaMemcpyDeviceToHost), "down");
            report("hcdown lo", T, 320, HK, check_out(lo, T, 320, X, Wd, 0, 320, HK, 0));
            report("hcdown inj", T, 4, HK, check_out(inj, T, 4, X, Wi, 0, 4, HK, 0));
        } else std::printf("hcdown T %lld: not run\n", (long long) T);
        ck(cudaMemset(dlo, 0xff, (size_t) T * 320 * 4), "poison");
        if (strata_pf_hcdown_exact_bf16(X.dev, HLD, Wd.dev, dlo, T, 320, HK, nullptr)) {
            any_ran = true;
            ck(cudaDeviceSynchronize(), "sync");
            std::vector<float> lo((size_t) T * 320);
            ck(cudaMemcpy(lo.data(), dlo, lo.size() * 4, cudaMemcpyDeviceToHost), "down");
            report("hcdown_exact", T, 320, HK, check_out(lo, T, 320, X, Wd, 0, 320, HK, 0));
        } else std::printf("hcdown_exact T %lld: not run\n", (long long) T);
        cudaFree(dlo);
        cudaFree(dinj);
        cudaFree(X.dev);
        cudaFree(Wd.dev);
        cudaFree(Wi.dev);
    }
    if (!any_ran) {
        std::printf("SKIP: no gfx11 WMMA device\n");
        return 77;
    }
    std::printf(fails == 0 ? "ALL OK\n" : "%d FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}
