// src/image/vae/qi21_pixels_check_main.cpp - test program strata-pixels-check (not shipped): qi21_float_to_u8 on a
// fixture of float values and the bytes stable-diffusion.cpp made of them.
//
//   strata-pixels-check FILE...     FILE: int32 n, then n float32 values, then n expected bytes
#include "strata/image/qi21_pixels.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

using namespace strata::image;

int main(int argc, char** argv) {
    int bad = 0;
    for (int i = 1; i < argc; ++i) {
        std::ifstream f(argv[i], std::ios::binary);
        int32_t n = 0;
        f.read((char*) &n, 4);
        std::vector<float> v(n > 0 ? n : 0);
        std::vector<uint8_t> want(v.size());
        f.read((char*) v.data(), 4 * (std::streamsize) v.size());
        f.read((char*) want.data(), (std::streamsize) want.size());
        if (!f || n <= 0) {
            std::printf("%s: cannot read\n", argv[i]);
            ++bad;
            continue;
        }
        size_t differ = 0;
        for (size_t k = 0; k < v.size(); ++k) differ += qi21_float_to_u8(v[k]) != want[k];
        std::printf("%s: %d values, %zu differ\n", argv[i], n, differ);
        bad += differ != 0;
    }
    std::printf("PIXELS-CHECK %s\n", bad || argc < 2 ? "FAIL" : "PASS");
    return bad || argc < 2 ? 1 : 0;
}
