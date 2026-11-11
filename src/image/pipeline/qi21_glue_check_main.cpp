// src/image/pipeline/qi21_glue_check_main.cpp - test program strata-glue-check (not shipped): the steps of
// include/strata/image/qi21_image.hpp on given inputs, for tools/test_qi21_glue.py (numpy float32 expectations) and
// for the check against stable-diffusion.cpp's own stage dumps.
//
//   strata-glue-check latent <noise.bin> <sigma0> <out.bin>
//   strata-glue-check cond <hidden.bin> <n> <hidden_dim> <weights.bin> <n_system> <out.bin>
//   strata-glue-check unit <in.bin> <out.bin>
//   strata-glue-check dumps <dump dir> [...]      noise.bin -> qi21_initial_latent(noise, 1.0) == dit_x_0.bin
//                                                 byte for byte (prints GLUE-DUMPS PASS / FAIL)
#include "strata/image/qi21_image.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace strata::image;

namespace {

bool read_f32(const std::string& p, std::vector<float>& v) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), {});
    v.resize(raw.size() / 4);
    std::memcpy(v.data(), raw.data(), v.size() * 4);
    return true;
}

bool write_f32(const std::string& p, const std::vector<float>& v) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), (std::streamsize) (v.size() * 4));
    return (bool) f;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    std::vector<float> a, w;
    if (mode == "latent" && argc == 5) {
        if (!read_f32(argv[2], a)) return 2;
        return write_f32(argv[4], qi21_initial_latent(a, std::strtof(argv[3], nullptr))) ? 0 : 2;
    }
    if (mode == "cond" && argc == 8) {
        if (!read_f32(argv[2], a) || !read_f32(argv[5], w)) return 2;
        const int n = std::atoi(argv[3]), hd = std::atoi(argv[4]), ns = std::atoi(argv[6]);
        if ((size_t) n * hd != a.size() || (size_t) n != w.size()) return 2;
        return write_f32(argv[7], qi21_condition(std::move(a), n, hd, w, ns)) ? 0 : 2;
    }
    if (mode == "unit" && argc == 4) {
        if (!read_f32(argv[2], a)) return 2;
        qi21_picture_to_unit(a.data(), a.size());
        return write_f32(argv[3], a) ? 0 : 2;
    }
    if (mode == "dumps" && argc >= 3) {
        int bad = 0;
        for (int i = 2; i < argc; ++i) {
            const std::string d = argv[i];
            std::vector<float> noise, x0;
            if (!read_f32(d + "/noise.bin", noise) || !read_f32(d + "/dit_x_0.bin", x0)) {
                std::printf("%s: dumps missing\n", d.c_str());
                ++bad;
                continue;
            }
            const std::vector<float> got = qi21_initial_latent(noise, 1.0f);
            size_t differ = got.size() == x0.size() ? 0 : got.size();
            for (size_t k = 0; k < got.size() && k < x0.size(); ++k) differ += std::memcmp(&got[k], &x0[k], 4) != 0;
            std::printf("%s: starting latent %s (%zu of %zu values differ)\n", d.c_str(),
                        differ ? "DIFFERS" : "byte-identical to dit_x_0", differ, x0.size());
            bad += differ != 0;
        }
        std::printf("GLUE-DUMPS %s\n", bad ? "FAIL" : "PASS");
        return bad ? 1 : 0;
    }
    std::fprintf(stderr, "usage: see the head of src/image/pipeline/qi21_glue_check_main.cpp\n");
    return 2;
}
