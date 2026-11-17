// src/image/noise/qi21_noise_check_main.cpp - test program strata-noise-check (not shipped): Strata's initial noise
// (include/strata/image/qi21_noise.hpp) against stable-diffusion.cpp's own noise dumps.
//
//   strata-noise-check <dump dir> [<dump dir> ...]     each dir dump_p<P>_s<SEED> with noise.bin (stage_hooks.py)
//   strata-noise-check --write SEED N OUT              our N values for SEED into OUT (float32), for the CPU test
//
// Prints per dir byte-identical or the count of differing values and the largest difference in ulps, and
// NOISE-CHECK PASS / FAIL.
#include "strata/image/qi21_noise.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace strata::image;

namespace {

int64_t key(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (u & 0x80000000u) ? -(int64_t) (u & 0x7fffffffu) : (int64_t) u;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 5 && std::strcmp(argv[1], "--write") == 0) {
        const std::vector<float> v = qi21_noise(std::strtoull(argv[2], nullptr, 10), (uint32_t) std::strtoul(argv[3], nullptr, 10));
        std::ofstream(argv[4], std::ios::binary).write(reinterpret_cast<const char*>(v.data()), v.size() * 4);
        return 0;
    }
    if (argc < 2) {
        std::fprintf(stderr, "usage: strata-noise-check <dump dir> [...] | --write SEED N OUT\n");
        return 2;
    }
    int bad = 0;
    for (int i = 1; i < argc; ++i) {
        const fs::path d(argv[i]);
        const std::string name = d.filename().string(), tag = d.parent_path().filename().string() + "/" + name;
        const size_t at = name.rfind("_s");
        std::ifstream f(d / "noise.bin", std::ios::binary);
        if (at == std::string::npos || !f) {
            std::printf("%s: no seed in the name or no noise.bin\n", tag.c_str());
            ++bad;
            continue;
        }
        const uint64_t seed = std::strtoull(name.c_str() + at + 2, nullptr, 10);
        std::vector<char> raw((std::istreambuf_iterator<char>(f)), {});
        std::vector<float> sd(raw.size() / 4);
        std::memcpy(sd.data(), raw.data(), sd.size() * 4);
        const std::vector<float> ours = qi21_noise(seed, (uint32_t) sd.size());
        size_t differ = 0;
        int64_t max_ulp = 0;
        for (size_t k = 0; k < sd.size(); ++k)
            if (std::memcmp(&sd[k], &ours[k], 4) != 0) {
                ++differ;
                max_ulp = std::max<int64_t>(max_ulp, std::llabs(key(sd[k]) - key(ours[k])));
            }
        std::printf("%s: seed %llu, %zu values: %s", tag.c_str(), (unsigned long long) seed, sd.size(),
                    differ == 0 ? "byte-identical" : "DIFFERS");
        if (differ) std::printf(" (%zu values, largest %lld ulp)", differ, (long long) max_ulp);
        std::printf("\n");
        bad += differ != 0;
    }
    std::printf("NOISE-CHECK %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
