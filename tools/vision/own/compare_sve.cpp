// tools/vision/own/compare_sve.cpp - compares two strata-vision output files (SVE1) of the same picture.
//
//   strata-vision-own-compare <a.sve> <b.sve>
// Prints one line:  n=<tokens> nx= ny= dim= global=<cosine of the two whole files as vectors> tokmin=<smallest per-token cosine>
//                   tokmean=<mean per-token cosine> maxabs=<largest |a-b|> rel_rms=<rms(a-b)/rms(a)>
// Exit 0 when both files read and have the same shape; 1 otherwise ("MISMATCH ..." printed).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct Sve {
    int32_t n = 0, nx = 0, ny = 0, dim = 0;
    std::vector<float> v;
};

static bool load(const char* path, Sve& s) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::printf("MISMATCH cannot open %s\n", path); return false; }
    int32_t h[5];
    if (std::fread(h, sizeof h, 1, f) != 1 || h[0] != 0x31455653 || h[1] <= 0 || h[4] <= 0) { std::printf("MISMATCH %s is not an SVE1 file\n", path); std::fclose(f); return false; }
    s.n = h[1]; s.nx = h[2]; s.ny = h[3]; s.dim = h[4];
    s.v.resize((size_t) s.n * s.dim);
    const bool ok = std::fread(s.v.data(), sizeof(float), s.v.size(), f) == s.v.size();
    std::fclose(f);
    if (!ok) std::printf("MISMATCH %s is cut short\n", path);
    return ok;
}

int main(int argc, char** argv) {
    int worst = 0;
    if (argc == 5 && std::strcmp(argv[3], "--worst") == 0) worst = std::atoi(argv[4]);
    else if (argc != 3) { std::fprintf(stderr, "usage: strata-vision-own-compare a.sve b.sve [--worst K]\n"); return 2; }
    Sve a, b;
    if (!load(argv[1], a) || !load(argv[2], b)) return 1;
    if (a.n != b.n || a.nx != b.nx || a.ny != b.ny || a.dim != b.dim) {
        std::printf("MISMATCH shapes differ: %d tokens (%dx%d, dim %d) vs %d tokens (%dx%d, dim %d)\n", a.n, a.nx, a.ny, a.dim, b.n, b.nx, b.ny, b.dim);
        return 1;
    }
    double ab = 0, aa = 0, bb = 0, dd = 0, maxabs = 0, tmin = 2.0, tsum = 0;
    struct Tok { int t; double cos, na, nb, maxd, dn; };
    std::vector<Tok> toks;
    for (int t = 0; t < a.n; ++t) {
        double tab = 0, taa = 0, tbb = 0, tdd = 0, tmax = 0;
        for (int i = 0; i < a.dim; ++i) {
            const double x = a.v[(size_t) t * a.dim + i], y = b.v[(size_t) t * a.dim + i];
            tab += x * y; taa += x * x; tbb += y * y;
            dd += (x - y) * (x - y);
            tdd += (x - y) * (x - y);
            maxabs = std::fmax(maxabs, std::fabs(x - y));
            tmax = std::fmax(tmax, std::fabs(x - y));
        }
        ab += tab; aa += taa; bb += tbb;
        const double c = taa > 0 && tbb > 0 ? tab / std::sqrt(taa * tbb) : 0.0;
        tmin = std::fmin(tmin, c);
        tsum += c;
        toks.push_back({t, c, std::sqrt(taa), std::sqrt(tbb), tmax, std::sqrt(tdd)});
    }
    std::printf("n=%d nx=%d ny=%d dim=%d global=%.6f tokmin=%.6f tokmean=%.6f maxabs=%.5g rel_rms=%.5g\n", a.n, a.nx, a.ny, a.dim,
                aa > 0 && bb > 0 ? ab / std::sqrt(aa * bb) : 0.0, tmin, tsum / a.n, maxabs, aa > 0 ? std::sqrt(dd / aa) : 0.0);
    if (worst > 0) {   // the tokens with the lowest cosine: where they sit, how large they are in each file, and how far apart
        std::sort(toks.begin(), toks.end(), [](const Tok& p, const Tok& q) { return p.cos < q.cos; });
        double mean_norm = std::sqrt(aa / a.n);
        std::printf("typical token norm in A: %.4g\n", mean_norm);
        for (int k = 0; k < worst && k < (int) toks.size(); ++k) {
            const Tok& w = toks[(size_t) k];
            std::printf("worst %d: token %d (x=%d y=%d of %dx%d) cos %.6f |A| %.5g |B| %.5g max|A-B| %.5g |A-B| %.5g\n", k + 1, w.t, w.t % a.nx, w.t / a.nx, a.nx,
                        a.ny, w.cos, w.na, w.nb, w.maxd, w.dn);
        }
    }
    return 0;
}
