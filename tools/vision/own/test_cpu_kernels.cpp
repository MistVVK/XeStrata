// tools/vision/own/test_cpu_kernels.cpp - every kernel tier of the CPU tower (the ones this CPU runs) against double-precision references, odd sizes and tails
// included, and the fork-join team (every task of every run exactly once).   strata-vision-own-cpu-test
#include "cpu_kernels.h"
#include "cpu_team.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata_vit;

static int g_fail = 0;
static void check(bool ok, const char* what, const char* tier, double worst, double tol) {
    if (!ok) {
        ++g_fail;
        std::printf("FAIL %-8s %s: worst %.3g (tolerance %.3g)\n", tier, what, worst, tol);
    }
}

static std::mt19937 rng(12345);
static float rnd(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }

static void test_gemm(const CpuOps& o) {
    const int Ms[] = {1, 5, 13, 49, 100}, Ks[] = {1, 7, 72, 256, 300}, Ns[] = {1, 8, 17, 33, 72, 100};
    double worst = 0, tol = 0;
    for (int M : Ms)
        for (int K : Ks)
            for (int N : Ns)
                for (int init = 0; init < 3; ++init) {
                    const int lda = K + 3, ldb = (N + 63) / 64 * 64 + 64, ldc = N + 5;
                    std::vector<float> A((size_t) M * lda), B((size_t) K * ldb + 128), C((size_t) M * ldc), C0, bias((size_t) N);
                    for (auto& v : A) v = rnd(-1, 1);
                    for (auto& v : B) v = rnd(-1, 1);
                    for (auto& v : C) v = rnd(-1, 1);
                    for (auto& v : bias) v = rnd(-1, 1);
                    C0 = C;
                    o.gemm(A.data(), (size_t) lda, M, B.data(), (size_t) ldb, K, N, C.data(), (size_t) ldc, init, init == 1 ? nullptr : bias.data());
                    for (int m = 0; m < M; ++m)
                        for (int n = 0; n < N; ++n) {
                            double s = init == 0 ? 0.0 : C0[(size_t) m * ldc + n];
                            if (init != 1) s += bias[(size_t) n];
                            for (int k = 0; k < K; ++k) s += (double) A[(size_t) m * lda + k] * B[(size_t) k * ldb + n];
                            worst = std::max(worst, std::fabs(s - C[(size_t) m * ldc + n]));
                        }
                    // columns past N and rows past M must be untouched
                    for (int m = 0; m < M; ++m)
                        for (int n = N; n < ldc; ++n)
                            if (C[(size_t) m * ldc + n] != C0[(size_t) m * ldc + n]) worst = 1e9;
                    tol = std::max(tol, 1e-5 * (K + 2));
                }
    check(worst <= tol, "gemm (odd M, K, N; init modes 0, 1, 2)", o.name, worst, tol);
}

static uint16_t rnd_f16bits() { return (uint16_t) ((rng() & 0x8000) | ((rng() % 28 + 2) << 10) | (rng() & 0x3ff)); }   // a finite normal half
static uint16_t rnd_bf16bits() { return (uint16_t) ((rng() & 0x8000) | ((rng() % 200 + 20) << 7) | (rng() & 0x7f)); }

static void test_pack_values(const CpuOps& o) {
    const int ldw = 300, rows = 40, K = 300;
    for (int kind = 0; kind <= 2; ++kind) {
        std::vector<uint16_t> w16((size_t) rows * ldw);
        std::vector<float> w32((size_t) rows * ldw);
        for (size_t i = 0; i < w16.size(); ++i) {
            w16[i] = kind == 2 ? rnd_f16bits() : rnd_bf16bits();
            w32[i] = rnd(-2, 2);
        }
        const void* W = kind == 0 ? (const void*) w32.data() : (const void*) w16.data();
        for (int nvalid : {o.NR, 5}) {
            const int n0 = 3, k0 = 17, kc = 200;
            std::vector<float> dst((size_t) kc * o.NR, -7.0f);
            o.pack_w(W, kind, (size_t) ldw, n0, nvalid, k0, kc, dst.data());
            double worst = 0;
            for (int k = 0; k < kc; ++k)
                for (int c = 0; c < o.NR; ++c) {
                    float want = 0.0f;
                    if (c < nvalid) {
                        const size_t idx = (size_t) (n0 + c) * ldw + (size_t) (k0 + k);
                        if (kind == 0) want = w32[idx];
                        else if (kind == 1) want = quant::bf16_to_f32(w16[idx]);
                        else want = quant::f16_to_f32(w16[idx]);
                    }
                    worst = std::max(worst, (double) std::fabs(want - dst[(size_t) k * o.NR + c]));
                }
            check(worst == 0.0, kind == 0 ? "pack_w float32" : kind == 1 ? "pack_w BF16" : "pack_w F16", o.name, worst, 0.0);
        }
    }
    (void) K;
}

static void test_gelu(const CpuOps& o) {
    const int rows = 3, n = 203, ld = 211;
    std::vector<float> x((size_t) rows * ld), y;
    for (auto& v : x) v = rnd(-12, 12);
    y = x;
    o.gelu(y.data(), (size_t) ld, rows, n);
    double worst = 0;
    for (int r = 0; r < rows; ++r)
        for (int i = 0; i < n; ++i) {
            const double v = x[(size_t) r * ld + i];
            const double want = 0.5 * v * (1.0 + std::tanh(0.7978845608028654 * v * (1.0 + 0.044715 * v * v)));
            worst = std::max(worst, std::fabs(want - y[(size_t) r * ld + i]) / std::max(1.0, std::fabs(v)));
        }
    for (int r = 0; r < rows; ++r)
        for (int i = n; i < ld; ++i)
            if (y[(size_t) r * ld + i] != x[(size_t) r * ld + i]) worst = 1e9;
    check(worst <= 3e-6, "gelu", o.name, worst, 3e-6);
}

static void test_layernorm(const CpuOps& o) {
    for (int D : {1152, 7, 72, 4608}) {
        std::vector<float> x((size_t) D), w((size_t) D), b((size_t) D), y((size_t) D);
        for (int i = 0; i < D; ++i) { x[(size_t) i] = rnd(-3, 5); w[(size_t) i] = rnd(0.5f, 1.5f); b[(size_t) i] = rnd(-1, 1); }
        o.layernorm(x.data(), y.data(), D, w.data(), b.data(), 1e-6f);
        double s = 0, q = 0, worst = 0;
        for (int i = 0; i < D; ++i) s += x[(size_t) i];
        const double mean = s / D;
        for (int i = 0; i < D; ++i) q += (x[(size_t) i] - mean) * (x[(size_t) i] - mean);
        const double scale = 1.0 / std::sqrt(q / D + 1e-6);
        for (int i = 0; i < D; ++i) worst = std::max(worst, std::fabs((x[(size_t) i] - mean) * scale * w[(size_t) i] + b[(size_t) i] - y[(size_t) i]));
        check(worst <= 2e-5, "layernorm", o.name, worst, 2e-5);
    }
}

static void test_add(const CpuOps& o) {
    for (size_t n : {1u, 7u, 8u, 33u, 1000u}) {
        std::vector<float> x(n), y(n), z;
        for (size_t i = 0; i < n; ++i) { x[i] = rnd(-1, 1); y[i] = rnd(-1, 1); }
        z = x;
        o.add(z.data(), y.data(), n);
        double worst = 0;
        for (size_t i = 0; i < n; ++i) worst = std::max(worst, (double) std::fabs(z[i] - (x[i] + y[i])));
        check(worst == 0.0, "add", o.name, worst, 0.0);
    }
}

// the online softmax over two key tiles, then P*V by the product kernel, against a plain softmax in double
static void test_attention(const CpuOps& o) {
    const int qr = 13, hd = 72, T1 = 100, T2 = 37, n = T1 + T2, lds = 256, ldo = 80, ldv = 3 * 1152;
    std::vector<float> S1((size_t) qr * lds), S2((size_t) qr * lds), Sfull((size_t) qr * n), V((size_t) n * ldv + 256), O((size_t) qr * ldo, 0.0f), m((size_t) qr, -3.0e38f), l((size_t) qr, 0.0f);
    for (int r = 0; r < qr; ++r)
        for (int j = 0; j < n; ++j) {
            const float s = rnd(-30, 30);
            Sfull[(size_t) r * n + j] = s;
            (j < T1 ? S1[(size_t) r * lds + j] : S2[(size_t) r * lds + (j - T1)]) = s;
        }
    for (auto& v : V) v = rnd(-1, 1);
    o.softmax_update(S1.data(), (size_t) lds, qr, T1, m.data(), l.data(), O.data(), (size_t) ldo, hd);
    o.gemm(S1.data(), (size_t) lds, qr, V.data(), (size_t) ldv, T1, hd, O.data(), (size_t) ldo, 1, nullptr);
    o.softmax_update(S2.data(), (size_t) lds, qr, T2, m.data(), l.data(), O.data(), (size_t) ldo, hd);
    o.gemm(S2.data(), (size_t) lds, qr, V.data() + (size_t) T1 * ldv, (size_t) ldv, T2, hd, O.data(), (size_t) ldo, 1, nullptr);
    double worst = 0;
    for (int r = 0; r < qr; ++r) {
        double mx = -1e300, den = 0;
        for (int j = 0; j < n; ++j) mx = std::max(mx, (double) Sfull[(size_t) r * n + j]);
        for (int j = 0; j < n; ++j) den += std::exp(Sfull[(size_t) r * n + j] - mx);
        for (int d = 0; d < hd; ++d) {
            double acc = 0;
            for (int j = 0; j < n; ++j) acc += std::exp(Sfull[(size_t) r * n + j] - mx) / den * V[(size_t) j * ldv + d];
            worst = std::max(worst, std::fabs(acc - O[(size_t) r * ldo + d] / l[(size_t) r]));
        }
    }
    check(worst <= 2e-5, "online softmax + P*V", o.name, worst, 2e-5);
}

static void test_team() {
    for (int nth : {1, 2, 7, 16}) {
        Team t(nth);
        bool ok = true;
        for (int run = 0; run < 300 && ok; ++run) {
            const int n = 1 + (run * 7) % 90;
            std::vector<std::atomic<int>> hits((size_t) n);
            for (auto& h : hits) h.store(0);
            std::atomic<int> bad_thread{0};
            t.run(n, [&](int task, int thr) {
                if (thr < 0 || thr >= nth) bad_thread.fetch_add(1);
                hits[(size_t) task].fetch_add(1);
            });
            if (bad_thread.load()) { ok = false; std::printf("team(%d) run %d: %d tasks saw a thread index out of range\n", nth, run, bad_thread.load()); }
            for (size_t i = 0; i < hits.size(); ++i)
                if (hits[i].load() != 1) { ok = false; std::printf("team(%d) run %d of %d tasks: task %zu ran %d times\n", nth, run, n, i, hits[i].load()); }
            if (run % 50 == 0) std::this_thread::sleep_for(std::chrono::milliseconds(2));      // let the workers go to sleep once in a while
        }
        check(ok, "team: every task once, thread index in range", "-", 0, 0);
    }
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "team") == 0) {     // only the fork-join team (a loop under ThreadSanitizer)
        const int reps = argc > 2 ? std::atoi(argv[2]) : 1;
        for (int i = 0; i < reps; ++i) test_team();
        std::printf(g_fail ? "%d FAILED\n" : "team: all passed\n", g_fail);
        return g_fail ? 1 : 0;
    }
    const quant::Caps& c = quant::caps();
    std::printf("cpu: avx2 %d fma %d f16c %d avx512 %d; best tier %s\n", c.avx2, c.fma, c.f16c, c.avx512, quant::isa_name(quant::best_isa()));
    std::vector<const CpuOps*> tiers = {&cpu_ops_scalar()};
    if (quant::isa_ok(quant::Isa::Avx2)) tiers.push_back(&cpu_ops_avx2());
    if (quant::isa_ok(quant::Isa::Avx512)) tiers.push_back(&cpu_ops_avx512());
    for (const CpuOps* o : tiers) {
        std::printf("tier %s (tile %d x %d)\n", o->name, o->MR, o->NR);
        test_gemm(*o);
        test_pack_values(*o);
        test_gelu(*o);
        test_layernorm(*o);
        test_add(*o);
        test_attention(*o);
    }
    test_team();
    std::printf(g_fail ? "%d FAILED\n" : "all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
