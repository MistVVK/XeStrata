// sy_ktest.cpp - unit tests of the lane lead's SYCL kernels against CPU references (linear_h for every trunk format, f32, split_kvb, rms, router, indexer pieces).
// build: HY4S_LIBS=0 b_sycl.sh <tag> sy_ktest sycl/sy_ktest.cpp -- <path>/engine/src/quant/sk_codecs.cpp
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "strata/quant/sk_codecs.hpp"
#include "sy_gemm_xmx.h"
#include "sy_indexer.h"
#include "sy_kernels.h"
#include "sy_linear.h"
#include "sy_rp.h"

using namespace hy4s;
using strata::quant::sk::SkCode;

static std::mt19937 rng(12345);
static float rnd(float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); }

static uint16_t f2h_bits(float f) { return sycl::bit_cast<uint16_t>((half)f); }

static void make_sk(int fmt, int N, int K, std::vector<uint8_t>& w) {
    const int sb = sk_sb_bytes(fmt);
    w.resize((size_t)N * (K / 256) * sb);
    for (auto& b : w) b = (uint8_t)(rng() & 255);
    for (size_t i = 0; i < w.size(); i += sb) {
        const uint16_t d = f2h_bits(rnd(0.001f, 0.05f));
        w[i] = d & 255;
        w[i + 1] = d >> 8;
        for (int g = 0; g < 4; ++g) w[i + 2 + g] = (uint8_t)(100 + (rng() % 150));
    }
}

static int fails = 0;
static void check(const char* what, const std::vector<float>& a, const std::vector<float>& ref, double tol_rel) {
    double num = 0, den = 0, mx = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double)a[i] - ref[i];
        num += d * d;
        den += (double)ref[i] * ref[i];
        mx = std::max(mx, std::fabs(d));
    }
    const double rel = std::sqrt(num / (den + 1e-30));
    printf("%-34s rel L2 %.3e  max abs %.3e  %s\n", what, rel, mx, rel <= tol_rel ? "ok" : "FAIL");
    if (rel > tol_rel) ++fails;
}

int main() {
    auto& q = Q();
    printf("device %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const int N = 300, K = 1536;
    for (int fmt : {FMT_S2, FMT_S3, FMT_S4, FMT_S5, FMT_S6, FMT_S8}) {
        std::vector<uint8_t> w;
        make_sk(fmt, N, K, w);
        for (int M : {1, 5, 20}) {
            std::vector<half> xh((size_t)M * K);
            std::vector<float> xf((size_t)M * K);
            for (size_t i = 0; i < xh.size(); ++i) {
                xh[i] = (half)rnd(-1.f, 1.f);
                xf[i] = (float)xh[i];
            }
            std::vector<float> ref((size_t)M * N);
            {
                std::vector<float> row(K);
                const size_t rb = (size_t)(K / 256) * sk_sb_bytes(fmt);
                for (int n = 0; n < N; ++n) {
                    strata::quant::sk::dequant_sk_row(w.data() + n * rb, (SkCode)fmt, row.data(), K);
                    for (int m = 0; m < M; ++m) {
                        double s = 0;
                        for (int k = 0; k < K; ++k) s += (double)row[k] * xf[(size_t)m * K + k];
                        ref[(size_t)m * N + n] = (float)s;
                    }
                }
            }
            uint8_t* dw = dalloc<uint8_t>(w.size());
            half* dx = dalloc<half>(xh.size());
            float* dy = dalloc<float>((size_t)M * N);
            h2d(dw, w.data(), w.size());
            h2d(dx, xh.data(), xh.size() * 2);
            linear_h(fmt, dw, N, K, dx, dy, M);
            std::vector<float> y((size_t)M * N);
            d2h(y.data(), dy, y.size() * 4);
            wait();
            char nm[64];
            snprintf(nm, sizeof nm, "linear_h S%d M=%d", fmt, M);
            check(nm, y, ref, 2e-6);
            dfree(dw); dfree(dx); dfree(dy);
        }
    }
    for (int fmt : {FMT_S4, FMT_S8}) {  // RP layout: repack on the host, GEMV (M <= 8), XMX GEMM (M >= 16, N % 64 == 0) vs the raw reference kernel
        for (int Nr : {320, 200}) {
            const int Kr = 1536;
            std::vector<uint8_t> w;
            make_sk(fmt, Nr, Kr, w);
            std::vector<uint8_t> rp(rp_bytes(fmt, Nr, Kr));
            rp_repack_host(fmt, w.data(), rp.data(), Nr, Kr);
            uint8_t* dw = dalloc<uint8_t>(w.size());
            uint8_t* dr = dalloc<uint8_t>(rp.size());
            h2d(dw, w.data(), w.size());
            h2d(dr, rp.data(), rp.size());
            for (int M : {1, 3, 8, 20, 77}) {
                std::vector<half> xh((size_t)M * Kr);
                for (auto& v : xh) v = (half)rnd(-1.f, 1.f);
                half* dx = dalloc<half>(xh.size());
                float* y1 = dalloc<float>((size_t)M * Nr);
                float* y2 = dalloc<float>((size_t)M * Nr);
                h2d(dx, xh.data(), xh.size() * 2);
                linear_h(fmt, dw, Nr, Kr, dx, y1, M);
                linear_auto(fmt, dr, Nr, Kr, dx, y2, M, true);
                std::vector<float> a((size_t)M * Nr), b((size_t)M * Nr);
                d2h(a.data(), y1, a.size() * 4);
                d2h(b.data(), y2, b.size() * 4);
                wait();
                char nm[96];
                snprintf(nm, sizeof nm, "RP S%d N=%d M=%d (%s)", fmt, Nr, M, (M >= 16 && !(Nr & 63)) ? "xmx" : "gemv");
                check(nm, b, a, (M >= 16 && !(Nr & 63)) ? 5e-4 : 2e-6);
                dfree(dx); dfree(y1); dfree(y2);
            }
            dfree(dw); dfree(dr);
        }
    }
    {  // Q8ROW and F32
        std::vector<uint8_t> w((size_t)N * 4 + (size_t)N * K);
        float* sc = (float*)w.data();
        for (int n = 0; n < N; ++n) sc[n] = rnd(0.001f, 0.02f);
        for (size_t i = (size_t)N * 4; i < w.size(); ++i) w[i] = (uint8_t)(rng() & 255);
        const int M = 7;
        std::vector<half> xh((size_t)M * K);
        for (auto& v : xh) v = (half)rnd(-1.f, 1.f);
        std::vector<float> ref((size_t)M * N);
        for (int n = 0; n < N; ++n)
            for (int m = 0; m < M; ++m) {
                double s = 0;
                for (int k = 0; k < K; ++k) s += (double)(int8_t)w[(size_t)N * 4 + (size_t)n * K + k] * (float)xh[(size_t)m * K + k];
                ref[(size_t)m * N + n] = (float)(s * sc[n]);
            }
        uint8_t* dw = dalloc<uint8_t>(w.size());
        half* dx = dalloc<half>(xh.size());
        float* dy = dalloc<float>((size_t)M * N);
        h2d(dw, w.data(), w.size());
        h2d(dx, xh.data(), xh.size() * 2);
        linear_h(FMT_Q8ROW, dw, N, K, dx, dy, M);
        std::vector<float> y((size_t)M * N);
        d2h(y.data(), dy, y.size() * 4);
        wait();
        check("linear_h Q8ROW M=7", y, ref, 2e-6);
    }
    {
        const int Nr = 512, Kr = 3072, M = 3;
        std::vector<float> w((size_t)Nr * Kr), x((size_t)M * Kr), ref((size_t)M * Nr);
        for (auto& v : w) v = rnd(-0.05f, 0.05f);
        for (auto& v : x) v = rnd(-1.f, 1.f);
        for (int n = 0; n < Nr; ++n)
            for (int m = 0; m < M; ++m) {
                double s = 0;
                for (int k = 0; k < Kr; ++k) s += (double)w[(size_t)n * Kr + k] * x[(size_t)m * Kr + k];
                ref[(size_t)m * Nr + n] = (float)s;
            }
        float* dw = dalloc<float>(w.size());
        float* dx = dalloc<float>(x.size());
        float* dy = dalloc<float>(ref.size());
        h2d(dw, w.data(), w.size() * 4);
        h2d(dx, x.data(), x.size() * 4);
        linear_f32(dw, Nr, Kr, dx, dy, M);
        std::vector<float> y(ref.size());
        d2h(y.data(), dy, y.size() * 4);
        wait();
        check("linear_f32 router M=3", y, ref, 1e-6);
        // router_topk
        std::vector<float> bias(512);
        for (auto& v : bias) v = rnd(-0.02f, 0.02f);
        float* db = dalloc<float>(512);
        int32_t* di = dalloc<int32_t>(M * 16);
        float* dwt = dalloc<float>(M * 16);
        h2d(db, bias.data(), 512 * 4);
        router_topk(dy, db, di, dwt, 512, 16, 3.998f, M);
        std::vector<int32_t> gi(M * 16);
        std::vector<float> gw(M * 16);
        d2h(gi.data(), di, M * 16 * 4);
        d2h(gw.data(), dwt, M * 16 * 4);
        wait();
        {  // the one-sub-group variant must give the same ids and weights
            int32_t* di2 = dalloc<int32_t>(M * 16);
            float* dwt2 = dalloc<float>(M * 16);
            router_topk_sg(dy, db, di2, dwt2, 16, 3.998f, M);
            std::vector<int32_t> gi2(M * 16);
            std::vector<float> gw2(M * 16);
            d2h(gi2.data(), di2, M * 16 * 4);
            d2h(gw2.data(), dwt2, M * 16 * 4);
            wait();
            int badsg = 0;
            double wsg = 0;
            for (int i = 0; i < M * 16; ++i) {
                if (gi2[i] != gi[i]) ++badsg;
                wsg = std::max(wsg, (double)std::fabs(gw2[i] - gw[i]));
            }
            printf("router_topk_sg vs router_topk: index mismatches %d, max weight diff %.2e  %s\n", badsg, wsg, (badsg == 0 && wsg < 1e-6) ? "ok" : "FAIL");
            if (badsg || wsg >= 1e-6) ++fails;
        }
        int bad = 0;
        double wmax = 0;
        for (int m = 0; m < M; ++m) {
            std::vector<float> p(512), s(512);
            std::vector<int> idx(512);
            for (int e = 0; e < 512; ++e) {
                p[e] = 1.f / (1.f + std::exp(-ref[(size_t)m * 512 + e]));
                s[e] = p[e] + bias[e];
                idx[e] = e;
            }
            std::partial_sort(idx.begin(), idx.begin() + 16, idx.end(), [&](int a, int b) { return s[a] > s[b] || (s[a] == s[b] && a < b); });
            float sum = 0;
            for (int k = 0; k < 16; ++k) sum += p[idx[k]];
            for (int k = 0; k < 16; ++k) {
                if (gi[m * 16 + k] != idx[k]) ++bad;
                wmax = std::max(wmax, (double)std::fabs(gw[m * 16 + k] - p[idx[k]] / (sum + 1e-20f) * 3.998f));
            }
        }
        printf("router_topk: index mismatches %d, max weight err %.2e  %s\n", bad, wmax, (bad == 0 && wmax < 1e-4) ? "ok" : "FAIL");
        if (bad || wmax >= 1e-4) ++fails;
    }
    {  // rms_rows + add_rms
        const int rows = 5, D = 3072;
        std::vector<float> x((size_t)rows * D), w(D), ref((size_t)rows * D);
        for (auto& v : x) v = rnd(-3.f, 3.f);
        for (auto& v : w) v = rnd(0.5f, 1.5f);
        for (int r = 0; r < rows; ++r) {
            double ss = 0;
            for (int i = 0; i < D; ++i) ss += (double)x[(size_t)r * D + i] * x[(size_t)r * D + i];
            const float inv = 1.0f / std::sqrt((float)(ss / D) + 1e-5f);
            for (int i = 0; i < D; ++i) ref[(size_t)r * D + i] = x[(size_t)r * D + i] * inv * w[i];
        }
        float* dx = dalloc<float>(x.size());
        float* dw = dalloc<float>(D);
        float* dy = dalloc<float>(x.size());
        h2d(dx, x.data(), x.size() * 4);
        h2d(dw, w.data(), D * 4);
        rms_rows(dx, dw, dy, nullptr, D, 1e-5f, rows, 256);
        std::vector<float> y(x.size());
        d2h(y.data(), dy, y.size() * 4);
        wait();
        check("rms_rows D=3072", y, ref, 1e-5);
        add_rms(dx, dx, dw, dy, D, 1e-5f, rows);
        d2h(y.data(), dy, y.size() * 4);
        wait();
        check("add_rms (a+a) D=3072", y, ref, 1e-5);
    }
    printf("%s (%d failures)\n", fails ? "SOME FAILED" : "ALL OK", fails);
    return fails ? 1 : 0;
}
