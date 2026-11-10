// ucrt_study.cpp - TEST-ONLY (OWN-5), Windows / MSVC only: what the Windows noise stream depends on.
// 1. sd.cpp's PhiloxRNG (its header, compiled by MSVC: unqualified log/sin pick logf/sinf) for seeds 42 and 1234,
//    with the CRT's FMA3 code paths on (default on an FMA CPU) and off (_set_FMA3_enable(0)).
// 2. Over every reachable u and v (83,886,081 each): how often the CRT's logf(u) / sinf(v) differ from the correctly
//    rounded float result (float(log_cr) / float(sin_cr) of own5_noise.hpp), FMA3 on and off.
//   ucrt_study <dir-with-core/rng_philox.hpp is an include path at build time> [threads]
#include <math.h>

#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "core/rng_philox.hpp"
#include "own5_noise.hpp"

using namespace own5noise;
static const uint64_t N_INT = 1ull << 24, F_LO = 0x4B800000ull, F_HI = 0x4F800000ull;
static const uint64_t N_DOM = N_INT + (F_HI - F_LO + 1);
static inline float x_of(uint64_t i) {
    if (i < N_INT) return (float) (uint32_t) i;
    uint32_t b = (uint32_t) (F_LO + (i - N_INT));
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}
static inline float u_of(float x) { volatile float p = (float) ((double) x * 0x1p-32); return p + 0x1p-33f; }
static inline float v_of(float y) {
    const float C = 0x1p-32f * 6.2831855f;
    volatile float p = (float) ((double) y * (double) C);
    return p + C / 2;
}
static uint64_t fnvb(const std::vector<float>& v) {
    const unsigned char* p = (const unsigned char*) v.data();
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < v.size() * 4; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

int main(int argc, char** argv) {
    const int T = argc > 1 ? std::atoi(argv[1]) : (int) std::thread::hardware_concurrency();
    std::printf("CRT FMA3 enabled by default: %d\n", _get_FMA3_enable());
    std::vector<float> crl(N_DOM), crs(N_DOM);     // correctly rounded float log(u), sin(v)
    {
        std::atomic<uint64_t> next{0};
        std::vector<std::thread> th;
        for (int t = 0; t < T; ++t) th.emplace_back([&] {
            for (uint64_t c; (c = next.fetch_add(1 << 16)) < N_DOM;)
                for (uint64_t i = c; i < c + (1 << 16) && i < N_DOM; ++i) {
                    const float x = x_of(i);
                    crl[i] = (float) log_cr(u_of(x));
                    crs[i] = (float) sin_cr(v_of(x));
                }
        });
        for (auto& t : th) t.join();
    }
    for (int fma = 1; fma >= 0; --fma) {
        _set_FMA3_enable(fma);
        for (uint64_t seed : {42ull, 1234ull}) {
            PhiloxRNG r;
            r.manual_seed(seed);
            std::vector<float> v = r.randn(262144);
            std::printf("FMA3 %d: sd.cpp MSVC stream seed %" PRIu64 " fnv64 %016" PRIx64 "\n", fma, seed, fnvb(v));
        }
        std::atomic<uint64_t> dl{0}, ds{0}, next{0};
        std::vector<std::thread> th;
        for (int t = 0; t < T; ++t) th.emplace_back([&] {
            uint64_t a = 0, b = 0;
            for (uint64_t c; (c = next.fetch_add(1 << 16)) < N_DOM;)
                for (uint64_t i = c; i < c + (1 << 16) && i < N_DOM; ++i) {
                    const float x = x_of(i);
                    a += fbits(logf(u_of(x))) != fbits(crl[i]);
                    b += fbits(sinf(v_of(x))) != fbits(crs[i]);
                }
            dl += a;
            ds += b;
        });
        for (auto& t : th) t.join();
        std::printf("FMA3 %d: logf != correctly rounded in %" PRIu64 " of %" PRIu64 " u; sinf != correctly rounded in %"
                    PRIu64 " of %" PRIu64 " v\n", fma, (uint64_t) dl, N_DOM, (uint64_t) ds, N_DOM);
    }
    return 0;
}
