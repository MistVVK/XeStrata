// seeds.cpp - TEST-ONLY (OWN-5): strata_noise.hpp against stable-diffusion.cpp's PhiloxRNG over many seeds.
//   seeds <first> <count> <n> [threads]       prints one line per seed: seed, fnv64(ours), fnv64(ref), equal
// Build with -DWITH_REF and -I<dir with sd.cpp's core/rng_philox.hpp> to include the reference; without it only
// our hashes are printed (for platforms whose own sd.cpp stream differs, compare with the x86 reference list).
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "strata_noise.hpp"
#ifdef WITH_REF
#include "core/rng_philox.hpp"
#endif

static uint64_t fnvb(const std::vector<float>& v) {
    const unsigned char* p = (const unsigned char*) v.data();
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < v.size() * 4; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

int main(int argc, char** argv) {
    std::fprintf(stderr, "sin exception table (compiled in): %zu entries\n", strata_noise::sin_exc().keys.size());
    const uint64_t first = std::strtoull(argv[1], nullptr, 10);
    const uint32_t count = (uint32_t) std::strtoul(argv[2], nullptr, 10);
    const uint32_t n = (uint32_t) std::strtoul(argv[3], nullptr, 10);
    const int T = argc > 4 ? std::atoi(argv[4]) : (int) std::thread::hardware_concurrency();
    std::vector<uint64_t> seeds;
    for (uint32_t i = 0; i < count; ++i) seeds.push_back(first + i);
    // edge seeds: the 2^31 random default range end, a 64-bit seed (both key words used), the API maximum 2^62
    for (uint64_t s : {2147483647ull, 4294967295ull, 4294967296ull, 0x123456789abcdefull, (1ull << 62)}) seeds.push_back(s);
    std::vector<uint64_t> ho(seeds.size()), hr(seeds.size());
    std::vector<uint32_t> ndiff(seeds.size());
    std::atomic<size_t> next{0};
    std::vector<std::thread> th;
    for (int t = 0; t < T; ++t) th.emplace_back([&] {
        for (size_t k; (k = next++) < seeds.size();) {
            std::vector<float> o = strata_noise::randn(seeds[k], n);
            ho[k] = fnvb(o);
#ifdef WITH_REF
            PhiloxRNG r;
            r.manual_seed(seeds[k]);
            std::vector<float> f = r.randn(n);
            hr[k] = fnvb(f);
            uint32_t d = 0;
            for (uint32_t i = 0; i < n; ++i) d += strata_noise::fbits(o[i]) != strata_noise::fbits(f[i]);
            ndiff[k] = d;
#endif
        }
    });
    for (auto& t : th) t.join();
    uint64_t all = 1469598103934665603ull, bad = 0;
    for (size_t k = 0; k < seeds.size(); ++k) {
        std::printf("%" PRIu64 " %016" PRIx64 " %016" PRIx64 " %u\n", seeds[k], ho[k], hr[k], ndiff[k]);
        for (int b = 0; b < 8; ++b) all = (all ^ ((ho[k] >> (8 * b)) & 0xff)) * 1099511628211ull;
        bad += ndiff[k] != 0;
    }
    std::printf("SEEDS %zu n %u hash-of-hashes %016" PRIx64 " seeds-with-differences %" PRIu64 "\n", seeds.size(), n, all, bad);
    return 0;
}
