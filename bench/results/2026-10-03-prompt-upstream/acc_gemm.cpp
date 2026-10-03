// Scratch probe: Gemm-style BF16 products with accumulate (xmx_gemm / gemm_rows) against hi-only + lo-only summed on
// the host, at the prompt path's BF16 projection shapes and odd T (edge tiles).
#include "strata/kernels/xmx_gemm.hpp"
#include "../../../src/kernels/parity_device.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
using namespace strata::kernels;
static uint16_t bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t) (u >> 16); }
int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    void* s = strata::parity::stream();
    struct S { int64_t N, K; };
    const S shapes[] = {{320, 10240}, {48, 2560}, {128, 2560}, {512, 2560}, {1, 2560}, {10240, 320}, {256, 2560}};
    const int64_t Ts[] = {1, 18, 100, 1500};
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0, 1);
    int fails = 0;
    for (const S& sh : shapes) for (int64_t T : Ts) {
        std::vector<uint16_t> a(T * sh.K), b(T * sh.K), w(sh.N * sh.K);
        for (auto& v : a) v = bf(nd(rng)); for (auto& v : b) v = bf(nd(rng) * 1e-3f); for (auto& v : w) v = bf(nd(rng) * 0.05f);
        strata::parity::Device d;
        auto *da = d.upload(a), *db = d.upload(b), *dw = d.upload(w);
        float* y = d.alloc<float>(T * sh.N); float* ya = d.alloc<float>(T * sh.N); float* yb = d.alloc<float>(T * sh.N);
        auto run = [&](const uint16_t* x, float* out, bool acc) {
            if (xmx_gemm_ok(x, dw, out, sh.N, sh.K, sh.N)) xmx_gemm(XmxType::bf16, x, dw, out, T, sh.N, sh.K, sh.N, s, acc);
            else gemm_rows(XmxType::bf16, x, dw, out, T, sh.N, sh.K, sh.N, s, acc);
        };
        run(da, ya, false); run(da, y, false); run(db, y, true);
        if (xmx_available(XmxType::bf16)) run(db, yb, false);
        else gemm_rows(XmxType::bf16, db, dw, yb, T, sh.N, sh.K, sh.N, s);   // the sum without XMX: gemm_rows
        std::vector<float> hy(T * sh.N), ha(T * sh.N), hb(T * sh.N);
        strata::parity::copy_out(hy.data(), y, hy.size()); strata::parity::copy_out(ha.data(), ya, ha.size());
        strata::parity::copy_out(hb.data(), yb, hb.size());
        double worst = 0;
        for (size_t i = 0; i < hy.size(); ++i) worst = std::fmax(worst, std::fabs((double) hy[i] - ((double) ha[i] + hb[i])) / (std::fabs(ha[i]) + 1e-3));
        const bool ok = worst < 1e-5;
        fails += !ok;
        std::printf("N %5lld K %5lld T %5lld %s worst rel %.2e %s\n", (long long) sh.N, (long long) sh.K, (long long) T,
                    xmx_gemm_ok(da, dw, y, sh.N, sh.K, sh.N) ? "xmx " : "rows", worst, ok ? "ok" : "FAIL");
    }
    return fails ? 1 : 0;
}
