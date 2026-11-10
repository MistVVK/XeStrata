// check_domain.cpp - TEST-ONLY (OWN-5): exhaustive check of own5_noise.hpp over every input Box-Muller can see.
// x = float(counter) for every 32-bit counter: the integers 0 .. 2^24-1 plus every float from 2^24 to 2^32
// (83,886,081 values).  From each x: u (log path) and, with x as y, v (sin path).
//   mode "glibc": compares log_cr / sin_cr / sqrt_cr with glibc's log / sin / sqrt (bit for bit) and the float s and
//                 the float r = s * sin as sd.cpp's x86 build makes them; lists every mismatch.
//   mode "hash":  only hashes our log and sin doubles over the domain (any platform, e.g. MSVC): equal hashes on two
//                 platforms = identical results for every reachable input, hence for every seed.
//   check_domain <glibc|hash> [threads]
#include <algorithm>
#include <atomic>
#include <utility>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "own5_noise.hpp"

using namespace own5noise;

static const uint64_t N_INT = 1ull << 24;
static const uint64_t F_LO = 0x4B800000ull, F_HI = 0x4F800000ull;
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
static inline uint64_t fnv(uint64_t h, uint64_t v) {
    for (int k = 0; k < 8; ++k) { h = (h ^ (v & 0xff)) * 1099511628211ull; v >>= 8; }
    return h;
}

int main(int argc, char** argv) {
    if (const char* t = std::getenv("OWN5_SIN_EXC"); t && *t) {        // the packed glibc exception table (sin_exc.bin)
        if (!own5noise::load_sin_exceptions(t)) { std::fprintf(stderr, "cannot read %s\n", t); return 2; }
        std::printf("sin exception table: %s, %zu entries\n", t, own5noise::sin_exc().keys.size());
    }
    const std::string mode = argc > 1 ? argv[1] : "hash";
    const int T = argc > 2 ? std::atoi(argv[2]) : (int) std::thread::hardware_concurrency();
    const int CH = 256;                                   // fixed chunking: the hash does not depend on T
    std::vector<uint64_t> hlog(CH), hsin(CH);
    std::atomic<int> next{0};
    std::atomic<uint64_t> mis_log{0}, mis_s{0}, mis_sin{0}, mis_sqrt{0}, mis_x86{0};
    std::mutex mu;
    std::vector<std::pair<uint32_t, uint64_t>> bad_log, bad_sin;   // input float bits, glibc double bits
    std::vector<std::thread> th;
    for (int t = 0; t < T; ++t) th.emplace_back([&] {
        for (int c; (c = next++) < CH;) {
            uint64_t a = N_DOM * c / CH, b = N_DOM * (c + 1) / CH, h1 = 1469598103934665603ull, h2 = h1;
            for (uint64_t i = a; i < b; ++i) {
                const float x = x_of(i), u = u_of(x), v = v_of(x);
                const double ol = log_cr(u), os = sin_cr(v);
                h1 = fnv(h1, dbits(ol));
                h2 = fnv(h2, dbits(sin_x86(v)));
                if (mode == "glibc") {
                    const double gl = std::log((double) u), gs = std::sin((double) v);
                    if (dbits(ol) != dbits(gl)) {
                        ++mis_log;
                        float s1 = ol == 0.0 ? -0.0f : (float) sqrt_cr(-2.0 * ol);
                        float s2 = (float) std::sqrt(-2.0f * gl);
                        if (fbits(s1) != fbits(s2)) ++mis_s;
                        std::lock_guard<std::mutex> g(mu);
                        bad_log.push_back({fbits(u), dbits(gl)});
                        if (mis_log < 20) std::printf("LOG u=%a ours=%a glibc=%a s %s\n", u, ol, gl,
                                                      fbits(s1) == fbits(s2) ? "same" : "DIFFERS");
                    }
                    if (gl != 0.0) {                      // our integer sqrt vs IEEE sqrt on glibc's own log
                        double q1 = sqrt_cr(-2.0 * gl), q2 = std::sqrt(-2.0 * gl);
                        if (dbits(q1) != dbits(q2)) ++mis_sqrt;
                    }
                    if (dbits(sin_x86(v)) != dbits(gs)) ++mis_x86;
                    if (dbits(os) != dbits(gs)) {
                        ++mis_sin;
                        std::lock_guard<std::mutex> g(mu);
                        bad_sin.push_back({fbits(v), dbits(gs)});
                        if (mis_sin < 20) std::printf("SIN v=%a ours=%a glibc=%a\n", v, os, gs);
                    }
                }
            }
            hlog[c] = h1;
            hsin[c] = h2;
        }
    });
    for (auto& t : th) t.join();
    uint64_t H1 = 1469598103934665603ull, H2 = H1;
    for (int c = 0; c < CH; ++c) { H1 = fnv(H1, hlog[c]); H2 = fnv(H2, hsin[c]); }
    std::printf("domain %" PRIu64 " values; hash(log_cr) %016" PRIx64 " hash(sin_x86) %016" PRIx64 "\n", N_DOM, H1, H2);
    if (mode == "glibc") {
        std::printf("log mismatches %" PRIu64 " (of which the float s differs: %" PRIu64 "), sin mismatches %" PRIu64
                    ", integer sqrt vs IEEE sqrt mismatches %" PRIu64
                    ", sin_x86 vs glibc mismatches (0 needs the exception table built in) %" PRIu64 "\n",
                    (uint64_t) mis_log, (uint64_t) mis_s, (uint64_t) mis_sin, (uint64_t) mis_sqrt, (uint64_t) mis_x86);
        // the exception lists: every reachable input where glibc differs from ours (float bits, glibc double bits)
        std::sort(bad_log.begin(), bad_log.end());
        std::sort(bad_sin.begin(), bad_sin.end());
        const std::pair<std::vector<std::pair<uint32_t, uint64_t>>*, const char*> lists[2] = {
            {&bad_log, "exc_log.txt"}, {&bad_sin, "exc_sin.txt"}};
        for (const auto& l : lists) {
            FILE* f = std::fopen(l.second, "w");
            for (const auto& e : *l.first) std::fprintf(f, "%08x %016" PRIx64 "\n", e.first, e.second);
            std::fclose(f);
        }
        std::printf("wrote exc_log.txt (%zu) and exc_sin.txt (%zu)\n", bad_log.size(), bad_sin.size());
    }
    return 0;
}
