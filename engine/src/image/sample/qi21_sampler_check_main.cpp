// src/image/sample/qi21_sampler_check_main.cpp - test program strata-sampler-check (not shipped): Strata's Turbo
// sampler (include/strata/image/qi21_sampler.hpp) against stable-diffusion.cpp's own stage dumps.
//
//   strata-sampler-check <dump dir> [<dump dir> ...] [--fixture OUT [--fixture-n N]]
//   strata-sampler-check --fixture-check FILE
//
// Dump mode: each dir holds one picture's dumps of all 8 model calls (tools/image/gate/stage_hooks.py: dit_x_<k>,
// dit_t_<k>, vel_<k>; and latent, the final latent).  Starting from dit_x_0 and feeding sd.cpp's own vel_<k>, our
// chain must give dit_x_<k+1> byte for byte for every step, and dit_t_<k> must equal qi21_timestep(sigma_k); the last
// step's result is compared with latent.bin when its size matches.  Prints per step the bytes-identical verdict and,
// where not identical, the count of differing values and the largest difference in units in the last place.
// --fixture OUT writes, from the first dir, N values per step (spread over the tensor) for the CPU test.
// Fixture mode: the step on the stored values must give the stored results (tools/test_qi21_sampler.py).
#include "strata/image/qi21_sampler.hpp"

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

bool read_f32(const fs::path& p, std::vector<float>& v) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), {});
    v.resize(raw.size() / 4);
    std::memcpy(v.data(), raw.data(), v.size() * 4);
    return true;
}

uint32_t bits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
}

// distance in units in the last place (signed-magnitude bits mapped to a monotonic integer line)
int64_t ulps(float a, float b) {
    auto key = [](float f) -> int64_t {
        const uint32_t u = bits(f);
        return (u & 0x80000000u) ? -(int64_t) (u & 0x7fffffffu) : (int64_t) u;
    };
    const int64_t d = key(a) - key(b);
    return d < 0 ? -d : d;
}

struct Cmp { size_t differ = 0; int64_t max_ulp = 0; };
Cmp compare(const std::vector<float>& a, const std::vector<float>& b) {
    Cmp c;
    for (size_t i = 0; i < a.size(); ++i)
        if (bits(a[i]) != bits(b[i])) {
            ++c.differ;
            c.max_ulp = std::max(c.max_ulp, ulps(a[i], b[i]));
        }
    return c;
}

const char MAGIC[4] = {'Q', 'S', '1', 0};

int fixture_check(const char* path) {
    std::ifstream f(path, std::ios::binary);
    char m[4];
    int32_t steps = 0, n = 0;
    if (!f.read(m, 4) || std::memcmp(m, MAGIC, 4) != 0 || !f.read((char*) &steps, 4) || !f.read((char*) &n, 4)) {
        std::printf("SAMPLER-FIXTURE FAIL (cannot read %s)\n", path);
        return 2;
    }
    const std::vector<float> sig = qi21_parse_sigmas(QI21_TURBO_SIGMAS);
    int bad = 0;
    for (int k = 0; k < steps; ++k) {
        float s0, s1;
        uint32_t tb;
        std::vector<float> x(n), v(n), want(n);
        f.read((char*) &s0, 4);
        f.read((char*) &s1, 4);
        f.read((char*) &tb, 4);
        f.read((char*) x.data(), 4 * n);
        f.read((char*) v.data(), 4 * n);
        f.read((char*) want.data(), 4 * n);
        if (!f) { std::printf("SAMPLER-FIXTURE FAIL (short file)\n"); return 2; }
        const bool sig_ok = k + 1 < (int) sig.size() && bits(sig[k]) == bits(s0) && bits(sig[k + 1]) == bits(s1);
        const bool t_ok = bits(qi21_timestep(s0)) == tb;
        qi21_euler_step(x.data(), v.data(), (size_t) n, s0, s1);
        const Cmp c = compare(x, want);
        std::printf("step %d: sigma %s, t %s, x_next %s", k, sig_ok ? "same" : "DIFFERS", t_ok ? "same" : "DIFFERS",
                    c.differ == 0 ? "identical" : "DIFFERS");
        if (c.differ) std::printf(" (%zu of %d values, up to %lld ulp)", c.differ, n, (long long) c.max_ulp);
        std::printf("\n");
        bad += !sig_ok || !t_ok || c.differ != 0;
    }
    std::printf("SAMPLER-FIXTURE %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "--fixture-check") == 0) return fixture_check(argv[2]);
    std::vector<fs::path> dirs;
    std::string fixture;
    int fixture_n = 512;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--fixture") == 0 && i + 1 < argc) fixture = argv[++i];
        else if (std::strcmp(argv[i], "--fixture-n") == 0 && i + 1 < argc) fixture_n = std::atoi(argv[++i]);
        else dirs.emplace_back(argv[i]);
    }
    if (dirs.empty()) {
        std::fprintf(stderr, "usage: strata-sampler-check <dump dir> [...] [--fixture OUT [--fixture-n N]] | --fixture-check FILE\n");
        return 2;
    }
    const std::vector<float> sig = qi21_parse_sigmas(QI21_TURBO_SIGMAS);
    const int steps = (int) sig.size() - 1;
    int bad = 0;
    std::ofstream fx;
    for (size_t di = 0; di < dirs.size(); ++di) {
        const fs::path& d = dirs[di];
        const std::string tag = d.parent_path().filename().string() + "/" + d.filename().string();
        std::vector<float> x;
        if (!read_f32(d / "dit_x_0.bin", x)) {
            std::printf("%s: dit_x_0 missing\n", tag.c_str());
            ++bad;
            continue;
        }
        const bool write_fx = di == 0 && !fixture.empty();
        if (write_fx) {
            fx.open(fixture, std::ios::binary);
            const int32_t st = steps - 1, n = fixture_n;   // steps 0..6: each has sd.cpp's next input to compare with
            fx.write(MAGIC, 4);
            fx.write((const char*) &st, 4);
            fx.write((const char*) &n, 4);
        }
        int first_bad = -1;
        for (int k = 0; k < steps; ++k) {
            std::vector<float> v, t, next;
            if (!read_f32(d / ("vel_" + std::to_string(k) + ".bin"), v) ||
                !read_f32(d / ("dit_t_" + std::to_string(k) + ".bin"), t) || v.size() != x.size() || t.empty()) {
                std::printf("%s step %d: dumps missing\n", tag.c_str(), k);
                ++bad;
                break;
            }
            const bool t_ok = bits(qi21_timestep(sig[k])) == bits(t[0]);
            const std::vector<float> xk = x;
            qi21_euler_step(x.data(), v.data(), x.size(), sig[k], sig[k + 1]);
            const bool last = k + 1 == steps;
            const fs::path want_p = last ? d / "latent.bin" : d / ("dit_x_" + std::to_string(k + 1) + ".bin");
            const bool have = read_f32(want_p, next) && next.size() == x.size();
            Cmp c;
            if (have) c = compare(x, next);
            std::printf("%s step %d: sigma %.6g -> %.6g, t %s (%.4f), %s %s", tag.c_str(), k, sig[k], sig[k + 1],
                        t_ok ? "identical" : "DIFFERS", t[0], last ? "latent.bin" : ("dit_x_" + std::to_string(k + 1)).c_str(),
                        !have ? "not comparable (missing or another size)" : c.differ == 0 ? "byte-identical" : "DIFFERS");
            if (have && c.differ)
                std::printf(" (%zu of %zu values, largest %lld ulp)", c.differ, x.size(), (long long) c.max_ulp);
            std::printf("\n");
            const bool step_bad = !t_ok || (!last && (!have || c.differ != 0));
            if (step_bad && first_bad < 0) first_bad = k;
            bad += step_bad;
            if (write_fx && !last) {
                const uint32_t tb = bits(t[0]);
                fx.write((const char*) &sig[k], 4);
                fx.write((const char*) &sig[k + 1], 4);
                fx.write((const char*) &tb, 4);
                const size_t stride = std::max<size_t>(1, x.size() / (size_t) fixture_n);
                const std::vector<float>* srcs[3] = {&xk, &v, &next};
                for (const std::vector<float>* src : srcs)
                    for (int i = 0; i < fixture_n; ++i) fx.write((const char*) &(*src)[(size_t) i * stride], 4);
            }
            if (!last && have) x = next;   // carry on from sd.cpp's own value, so one bad step does not hide the next
        }
        std::printf("%s: %s\n", tag.c_str(), first_bad < 0 ? "every step byte-identical"
                                                           : ("first differing step " + std::to_string(first_bad)).c_str());
    }
    std::printf("SAMPLER-CHECK %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
