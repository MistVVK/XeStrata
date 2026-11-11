// src/image/vae/qi21_vae_io_check_main.cpp - test program strata-vae-io-check (not shipped): the two byte-comparable
// steps around the VAE decoder against stable-diffusion.cpp's own stage dumps (tools/image/gate/stage_hooks.py):
//   latent.bin -> wan_vae_denormalize -> must equal vae_in.bin (what sd.cpp's VAE decoder got);
//   vae_out.bin (sd.cpp's decoded float picture) -> qi21_picture_to_u8 -> written to <out dir>/<tag>.rgb for the
//   comparison with the pixels of sd-cli's own PNG (tools/image/gate/png_pixels.py); a fused-multiply-add variant
//   (v * 255 + 0.5 in one rounding, how GCC may compile sd.cpp on aarch64) is written beside it as <tag>.fma.rgb.
//
//   strata-vae-io-check <out dir> <dump dir> [<dump dir> ...]
#include "strata/image/qi21_pixels.hpp"
#include "strata/image/wan_vae.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace strata::image;

namespace {

bool read_f32(const fs::path& p, std::vector<float>& v) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), {});
    v.resize(raw.size() / 4);
    std::memcpy(v.data(), raw.data(), v.size() * 4);
    return true;
}

std::vector<long long> read_shape(const fs::path& p) {
    std::vector<long long> s;
    std::ifstream f(p);
    long long x;
    while (f >> x) s.push_back(x);
    return s;
}

uint8_t fused_u8(float v) {   // the aarch64 GCC reading of sd.cpp's expression
    if (v <= 0.0f) return 0;
    if (v >= 1.0f) return 255;
    return static_cast<uint8_t>(std::fmaf(v, 255.0f, 0.5f));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: strata-vae-io-check <out dir> <dump dir> [...]\n");
        return 2;
    }
    const fs::path out(argv[1]);
    fs::create_directories(out);
    int bad = 0;
    for (int i = 2; i < argc; ++i) {
        const fs::path d(argv[i]);
        const std::string tag = d.parent_path().filename().string() + "__" + d.filename().string();
        std::vector<float> lat, vin, vout;
        const auto ls = read_shape(d / "latent.shape"), os = read_shape(d / "vae_out.shape");
        if (!read_f32(d / "latent.bin", lat) || !read_f32(d / "vae_in.bin", vin) || !read_f32(d / "vae_out.bin", vout) ||
            ls.size() < 3 || os.size() < 3) {
            std::printf("%s: dumps missing\n", tag.c_str());
            ++bad;
            continue;
        }
        const int lw = (int) ls[0], lh = (int) ls[1];
        wan_vae_denormalize(lat.data(), lh, lw);
        size_t differ = 0;
        for (size_t k = 0; k < lat.size() && k < vin.size(); ++k) differ += std::memcmp(&lat[k], &vin[k], 4) != 0;
        const bool in_ok = lat.size() == vin.size() && differ == 0;
        std::printf("%s: VAE input (%d x %d x %lld) %s", tag.c_str(), lw, lh, (long long) ls[2],
                    in_ok ? "byte-identical" : "DIFFERS");
        if (!in_ok) std::printf(" (%zu of %zu values; sizes %zu / %zu)", differ, vin.size(), lat.size(), vin.size());
        std::printf("\n");
        bad += !in_ok;
        const int w = (int) os[0], h = (int) os[1], c = (int) os[2];
        std::vector<uint8_t> px((size_t) w * h * c), fx((size_t) w * h * c);
        qi21_picture_to_u8(vout.data(), w, h, c, px.data());
        const size_t np = (size_t) w * h;
        for (size_t p = 0; p < np; ++p)
            for (int k = 0; k < c; ++k) fx[p * c + k] = fused_u8(vout[(size_t) k * np + p]);
        std::ofstream(out / (tag + ".rgb"), std::ios::binary).write((const char*) px.data(), px.size());
        std::ofstream(out / (tag + ".fma.rgb"), std::ios::binary).write((const char*) fx.data(), fx.size());
        size_t fd = 0;
        for (size_t k = 0; k < px.size(); ++k) fd += px[k] != fx[k];
        std::printf("%s: picture %d x %d x %d -> %s.rgb (separate roundings) and .fma.rgb (fused): %zu bytes differ "
                    "between the two\n", tag.c_str(), w, h, c, tag.c_str(), fd);
    }
    std::printf("VAE-IO-CHECK (VAE input) %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
