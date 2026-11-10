// src/image/vae/wan_vae_decode_main.cpp - test program for Strata's VAE decoder (OWN-5 gate harness).
//
//   strata-vae-decode <vae.safetensors> <latent.bin> <out.f32> [--dump DIR] [--repeat N] [--device D] [--stream on|off]
//
// latent.bin: the diffusion-space latent as run_set.py --dump writes it (float32 planar [64][h][w], latent.shape beside
// it says "w h 64 1"); it is denormalized like sd.cpp's decode_first_stage does, then decoded.  out.f32: float32
// planar [4][16h][16w] (before sd.cpp's 0-1 scaling; tools/image/gate/vae_ref.py --png makes the picture).
// --dump DIR writes every stage as DIR/<name>.f32 (pixel-major [H][W][C]) with DIR/<name>.shape "H W C".
// --stream off: the one-pass reference decode (the product always uses the row bands; same bytes, more memory).
// --repeat N decodes N times and checks that every run gives the same bytes (run to run), printing each time.
// Prints one line per run: "decode <ms> ms, peak <MiB> MiB (weights+work <MiB>)".
#include "strata/image/wan_vae.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace strata::image;

static bool read_file(const std::string& p, std::vector<float>& v, size_t n) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return false;
    v.resize(n);
    size_t got = std::fread(v.data(), 4, n, f);
    std::fclose(f);
    return got == n;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: strata-vae-decode <vae.safetensors> <latent.bin> <out.f32> [--dump DIR] [--repeat N] [--device D]\n");
        return 2;
    }
    std::string vae = argv[1], lat = argv[2], outp = argv[3], dump_dir;
    int repeat = 1, device = -1, stream = 1;
    for (int i = 4; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--dump")) dump_dir = argv[i + 1];
        else if (!std::strcmp(argv[i], "--repeat")) repeat = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--device")) device = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--stream")) stream = !std::strcmp(argv[i + 1], "on");
    }
    std::string shp = lat.substr(0, lat.size() - 4) + ".shape";
    std::ifstream sf(shp);
    int w = 0, h = 0, c = 0;
    if (!(sf >> w >> h >> c) || c != 64) {
        std::fprintf(stderr, "cannot read %s (expected \"w h 64 1\")\n", shp.c_str());
        return 2;
    }
    std::vector<float> latent;
    if (!read_file(lat, latent, (size_t) 64 * h * w)) {
        std::fprintf(stderr, "cannot read %s\n", lat.c_str());
        return 2;
    }
    wan_vae_denormalize(latent.data(), h, w);
    std::string err;
    const auto t_open = std::chrono::steady_clock::now();
    WanVae* v = wan_vae_open(vae, device, &err);
    if (!v) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 3;
    }
    std::printf("open %.1f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_open).count());
    wan_vae_set_streaming(v, stream != 0);
    if (!dump_dir.empty()) {
        wan_vae_set_dump(v, [&](const char* name, const float* data, int H, int W, int C) {
            std::string b = dump_dir + "/" + name;
            if (FILE* f = std::fopen((b + ".f32").c_str(), "wb")) { std::fwrite(data, 4, (size_t) H * W * C, f); std::fclose(f); }
            if (FILE* f = std::fopen((b + ".shape").c_str(), "w")) { std::fprintf(f, "%d %d %d\n", H, W, C); std::fclose(f); }
        });
    }
    const size_t n_out = (size_t) 4 * 16 * h * 16 * w;
    std::vector<float> out(n_out), first;
    int rc = 0;
    for (int r = 0; r < repeat; ++r) {
        WanVaeStats st;
        err = wan_vae_decode(v, latent.data(), h, w, out.data(), &st);
        if (!err.empty()) {
            std::fprintf(stderr, "%s\n", err.c_str());
            rc = 4;
            break;
        }
        bool same = true;
        if (r == 0) first = out;
        else same = std::memcmp(first.data(), out.data(), n_out * 4) == 0;
        std::printf("decode %dx%d run %d: %.1f ms, peak %.0f MiB (weights+work %.0f MiB)%s\n", 16 * w, 16 * h, r,
                    st.decode_ms, st.peak_bytes / 1048576.0, st.weight_bytes / 1048576.0,
                    r == 0 ? "" : (same ? ", identical to run 0" : ", DIFFERS from run 0"));
        if (!same) rc = 5;
        if (r == 0) wan_vae_set_dump(v, nullptr);
    }
    if (rc == 0 || rc == 5) {
        FILE* f = std::fopen(outp.c_str(), "wb");
        if (!f || std::fwrite(first.data(), 4, n_out, f) != n_out) rc = 6;
        if (f) std::fclose(f);
    }
    wan_vae_close(v);
    return rc;
}
