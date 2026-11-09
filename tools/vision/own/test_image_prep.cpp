// tools/vision/own/test_image_prep.cpp - own_prep_check: our picture preparation against mtmd's, bit for bit.
//
//   strata-vision-own-prep-check <mmproj.gguf> <oracle dir> <min_tokens> <max_tokens> <image> [<image> ...]
// The oracle files come from strata-vision-prep-oracle (oracle_prep.cpp) run with the same limits.  For every image: our decode, smart_resize,
// bicubic resize with padding and normalisation must reproduce mtmd's float buffer exactly (same size, every value equal).
// Without images it only runs the self-checks of smart_resize.
#include "image_prep.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace strata_vit;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static std::string base_name(const std::string& p) {
    const size_t s = p.find_last_of("/\\");
    return s == std::string::npos ? p : p.substr(s + 1);
}

int main(int argc, char** argv) {
    // --- smart_resize self-checks (the 32-aligned sides and the pixel bounds) ---
    {
        const int align = 32, mn = 8 * 1024, mx = 4096 * 1024;
        for (int w : {1, 2, 31, 33, 64, 100, 640, 800, 1000, 1024, 3000, 4000, 8000, 20000})
            for (int h : {1, 3, 32, 45, 90, 400, 450, 517, 1024, 3000, 8000}) {
                const Size2 s = smart_resize(w, h, align, mn, mx);
                CHECK(s.w >= align && s.h >= align && s.w % align == 0 && s.h % align == 0);
                CHECK((long long) s.w * s.h <= (long long) mx + 2LL * align * (s.w + s.h));   // the floor of each side can only shrink it
                if ((long long) w * h * 4 >= mn && (long long) w * h <= mx / 2) CHECK((long long) s.w * s.h >= mn / 2);
            }
        CHECK(smart_resize(640, 400, 32, 8192, 4194304) == (Size2{640, 416}));
        CHECK(smart_resize(1024, 1024, 32, 8192, 4194304) == (Size2{1024, 1024}));
        CHECK(smart_resize(64, 48, 32, 8192, 4194304).w * smart_resize(64, 48, 32, 8192, 4194304).h >= 8192);
        CHECK(smart_resize(0, 5, 32, 8192, 4194304) == (Size2{0, 0}));
    }
    if (argc < 6) {
        std::printf(g_fail ? "own_prep_check: %d FAILED\n" : "own_prep_check: self-checks ok\n", g_fail);
        return g_fail ? 1 : 0;
    }

    const std::string mmproj = argv[1], dir = argv[2];
    const int min_t = std::atoi(argv[3]), max_t = std::atoi(argv[4]);
    Gguf gg;
    std::string err;
    VitGeometry geo;
    if (!gg.open(mmproj, err) || !vit_geometry_from_gguf(gg, geo, err)) { std::printf("FAIL %s: %s\n", mmproj.c_str(), err.c_str()); return 1; }

    int same = 0, total = 0;
    for (int a = 5; a < argc; ++a) {
        const std::string img = argv[a];
        ++total;
        ImageU8 src, resized;
        PreparedImage prep;
        if (!image_load_file(img, src, err) || !prepare_image(src, geo, min_t, max_t, prep, err, &resized)) {
            std::printf("FAIL %s: %s\n", img.c_str(), err.c_str());
            ++g_fail;
            continue;
        }
        const std::string path = dir + "/" + base_name(img) + "." + std::to_string(min_t) + "." + std::to_string(max_t) + ".prep";
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) { std::printf("FAIL %s: no oracle file %s\n", img.c_str(), path.c_str()); ++g_fail; continue; }
        int32_t hdr[2] = {0, 0};
        if (std::fread(hdr, sizeof hdr, 1, f) != 1) { std::printf("FAIL %s: oracle header\n", img.c_str()); ++g_fail; std::fclose(f); continue; }
        std::vector<float> ref((size_t) hdr[0] * hdr[1] * 3);
        const bool got = std::fread(ref.data(), sizeof(float), ref.size(), f) == ref.size();
        std::fclose(f);
        if (!got) { std::printf("FAIL %s: oracle data short\n", img.c_str()); ++g_fail; continue; }
        if (hdr[0] != prep.w || hdr[1] != prep.h) {
            std::printf("FAIL %s: %dx%d -> ours %dx%d, mtmd %dx%d\n", img.c_str(), src.w, src.h, prep.w, prep.h, hdr[0], hdr[1]);
            ++g_fail;
            continue;
        }
        const size_t n = (size_t) prep.w * prep.h;
        size_t differ = 0;
        double maxd = 0;
        for (size_t i = 0; i < n; ++i)
            for (int c = 0; c < 3; ++c) {
                const float x = prep.chw[c * n + i], y = ref[i * 3 + c];
                if (std::memcmp(&x, &y, 4) != 0) { ++differ; maxd = std::fmax(maxd, std::fabs((double) x - y)); }
            }
        std::printf("%-28s %5dx%-5d -> %4dx%-4d (%d x %d tokens)  %s", base_name(img).c_str(), src.w, src.h, prep.w, prep.h, prep.w / geo.align(),
                    prep.h / geo.align(), differ ? "DIFFERS" : "identical");
        if (differ) std::printf(": %zu of %zu values, max |diff| %g", differ, n * 3, maxd);
        std::printf("\n");
        if (differ) ++g_fail; else ++same;
    }
    std::printf("own_prep_check: %d of %d pictures identical to mtmd (min_tokens %d, max_tokens %d)%s\n", same, total, min_t, max_t, g_fail ? ", FAILED" : "");
    return g_fail ? 1 : 0;
}
