// src/image/dit/qi21_dit_step_main.cpp - test program strata-dit-step (not shipped): one Qwen-Image-2.1 DiT step in
// Strata's own code (include/strata/image/qi21_dit.hpp) on the inputs stable-diffusion.cpp gave its DiT.
//
//   strata-dit-step <dit.gguf> <out dir> <dump dir> [<dump dir> ...] [--repeat N] [--call K] [--f16-resident]
//
// Each dump dir holds the stage-hook dumps of one picture (tools/image/gate/stage_hooks.py): dit_x_<K> (the latent,
// shape "W H C 1"), dit_t_<K> (the timestep) and cond (shape "context L").  Writes <out dir>/<dump parent name>/<dump name>/vel.bin, the
// velocity as sd.cpp's vel_<K>.bin ([C][H][W] float32), and vel.shape.  With --repeat N the step runs N times on the
// same inputs (the first runs the text prefix, the rest reuse it, like sampling steps 2..8) and every run must give
// the same bytes.  Prints the open report, per run the step and prefix time, the memory peak of the DiT's own buffers
// against the GGUF size, and DIT-STEP PASS / FAIL (FAIL = an error or runs that differ).
#include "strata/image/qi21_dit.hpp"

#include <algorithm>
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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: strata-dit-step <dit.gguf> <out dir> <dump dir> [...] [--repeat N] [--call K]\n");
        return 2;
    }
    int repeat = 1, call = 0, flags = 0;
    std::vector<fs::path> dumps;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) repeat = std::max(1, std::atoi(argv[++i]));
        else if (std::strcmp(argv[i], "--call") == 0 && i + 1 < argc) call = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--f16-resident") == 0) flags |= QI21_DIT_F16_RESIDENT;
        else dumps.emplace_back(argv[i]);
    }
    const unsigned long long gguf_bytes = fs::file_size(argv[1]);
    std::string err;
    Qi21Dit* dit = qi21_dit_open(argv[1], -1, &err, flags);
    if (!dit) {
        std::fprintf(stderr, "%s\n", err.c_str());
        std::printf("DIT-STEP FAIL\n");
        return 2;
    }
    std::printf("open: %s\n", qi21_dit_open_report(dit).c_str());
    int bad = 0;
    size_t peak = 0;
    for (const auto& d : dumps) {
        const std::string tag = d.parent_path().filename().string() + "/" + d.filename().string();
        const std::string xs = "dit_x_" + std::to_string(call), ts = "dit_t_" + std::to_string(call);
        std::vector<float> x, t, cond;
        const auto sx = read_shape(d / (xs + ".shape")), sc = read_shape(d / "cond.shape");
        if (!read_f32(d / (xs + ".bin"), x) || !read_f32(d / (ts + ".bin"), t) || !read_f32(d / "cond.bin", cond) ||
            sx.size() < 3 || sc.size() < 2 || t.empty()) {
            std::printf("%s: inputs missing\n", tag.c_str());
            ++bad;
            continue;
        }
        const int W = (int) sx[0], H = (int) sx[1], C = (int) sx[2], ctx = (int) sc[0], L = (int) sc[1];
        if (C != qi21_dit_in_channels(dit) || ctx != qi21_dit_context_dim(dit) || x.size() != (size_t) W * H * C ||
            cond.size() != (size_t) ctx * L) {
            std::printf("%s: shapes do not match the model\n", tag.c_str());
            ++bad;
            continue;
        }
        const int oc = qi21_dit_out_channels(dit);
        std::vector<float> vel((size_t) oc * W * H), first;
        std::vector<double> later;   // step times of runs 2.. (the text prefix reused, as in sampling steps 2..8)
        for (int r = 0; r < repeat; ++r) {
            Qi21DitStats st;
            err = qi21_dit_step(dit, x.data(), W, H, cond.data(), L, t[0], vel.data(), &st);
            if (!err.empty()) {
                std::printf("%s: %s\n", tag.c_str(), err.c_str());
                ++bad;
                break;
            }
            peak = std::max(peak, st.peak_bytes);
            if (r > 0) later.push_back(st.step_ms);
            std::printf("%s run %d: %dx%d latent, %d text tokens, t %.4f: step %.1f ms%s; weights %.3f GB, work %.3f GB, "
                        "peak %.3f GB (GGUF %.3f GB)\n",
                        tag.c_str(), r + 1, W, H, L, t[0], st.step_ms,
                        st.prefix_ran ? (" (text prefix " + std::to_string((int) st.prefix_ms) + " ms of it)").c_str() : "",
                        st.weight_bytes / 1e9, st.work_bytes / 1e9, st.peak_bytes / 1e9, gguf_bytes / 1e9);
            if (r == 0) {
                first = vel;
            } else if (std::memcmp(first.data(), vel.data(), vel.size() * 4) != 0) {
                std::printf("%s run %d: DIFFERS from run 1\n", tag.c_str(), r + 1);
                ++bad;
            }
        }
        if (!later.empty()) {
            std::sort(later.begin(), later.end());
            std::printf("%s: median step of runs 2..%d: %.1f ms (min %.1f, max %.1f)\n", tag.c_str(), repeat,
                        later[later.size() / 2], later.front(), later.back());
        }
        const fs::path od = fs::path(argv[2]) / d.parent_path().filename() / d.filename();
        fs::create_directories(od);
        std::ofstream(od / "vel.bin", std::ios::binary).write(reinterpret_cast<const char*>(first.data()), first.size() * 4);
        std::ofstream(od / "vel.shape") << W << " " << H << " " << oc << " 1\n";
    }
    std::printf("peak of the DiT's own device memory: %.3f GB (GGUF %.3f GB)\n", peak / 1e9, gguf_bytes / 1e9);
    qi21_dit_close(dit);
    std::printf("DIT-STEP %s\n", bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
