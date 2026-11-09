// tools/vision/own/oracle_prep.cpp - mtmd's side of the picture-preparation check (built in tools/vision's mtmd build only, EXCLUDE_FROM_ALL).
//
//   strata-vision-prep-oracle <mmproj.gguf> <out dir> <min_tokens> <max_tokens> <image> [<image> ...]
// For each image: decodes it with mtmd's own helper, runs mtmd's Qwen preprocessor (the same class mtmd_tokenize uses) with the given token
// limits (0 = the tower's default), and writes <out dir>/<image file name>.<min>.<max>.prep:  int32 {w, h}, then float32 [h][w][3] (what the
// tower is given: resized, padded, normalised).  `own_prep_check` (test_image_prep.cpp) compares its own result with these files, bit for bit.
#include "clip.h"
#include "clip-impl.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "mtmd-image.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string base_name(const std::string& p) {
    const size_t s = p.find_last_of("/\\");
    return s == std::string::npos ? p : p.substr(s + 1);
}

int main(int argc, char** argv) {
    if (argc < 6) { std::fprintf(stderr, "usage: strata-vision-prep-oracle <mmproj> <out dir> <min_tokens> <max_tokens> <image>...\n"); return 2; }
    const std::string mmproj = argv[1], out_dir = argv[2];
    const int min_t = std::atoi(argv[3]), max_t = std::atoi(argv[4]);

    clip_context_params cp{};
    cp.use_gpu = false;
    cp.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
    cp.image_min_tokens = min_t > 0 ? min_t : -1;
    cp.image_max_tokens = max_t > 0 ? max_t : -1;
    cp.warmup = false;
    clip_init_result r = clip_init(mmproj.c_str(), cp);
    if (!r.ctx_v) { std::fprintf(stderr, "cannot load the vision encoder %s\n", mmproj.c_str()); return 1; }
    mtmd_image_preprocessor_dyn_size pre(r.ctx_v);

    // a bitmap decoder context is not needed for stb images: the helper takes a null mtmd_context only for audio/video, which these are not
    int rc = 0;
    for (int a = 5; a < argc; ++a) {
        const std::string img = argv[a];
        // mtmd's own decode (stb_image, 3 channels)
        FILE* f = std::fopen(img.c_str(), "rb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", img.c_str()); rc = 1; continue; }
        std::vector<unsigned char> buf;
        unsigned char tmp[1 << 16];
        size_t k;
        while ((k = std::fread(tmp, 1, sizeof tmp, f)) > 0) buf.insert(buf.end(), tmp, tmp + k);
        std::fclose(f);
        mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_buf(nullptr, buf.data(), buf.size(), false, mtmd_helper_init_opt_default());
        if (!bw.bitmap) { std::fprintf(stderr, "cannot decode %s\n", img.c_str()); rc = 1; continue; }
        clip_image_u8 u8;
        u8.set_size({(int) mtmd_bitmap_get_nx(bw.bitmap), (int) mtmd_bitmap_get_ny(bw.bitmap)}, false);
        u8.cpy_buf(std::vector<uint8_t>(mtmd_bitmap_get_data(bw.bitmap), mtmd_bitmap_get_data(bw.bitmap) + mtmd_bitmap_get_n_bytes(bw.bitmap)));
        mtmd_image_preproc_out out = pre.preprocess(u8);
        if (out.entries.size() != 1) { std::fprintf(stderr, "%s: %zu entries\n", img.c_str(), out.entries.size()); rc = 1; continue; }
        const clip_image_f32& e = out.entries[0];
        const std::string path = out_dir + "/" + base_name(img) + "." + std::to_string(min_t) + "." + std::to_string(max_t) + ".prep";
        FILE* o = std::fopen(path.c_str(), "wb");
        if (!o) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); rc = 1; continue; }
        const int32_t hdr[2] = {e.nx(), e.ny()};
        std::fwrite(hdr, sizeof hdr, 1, o);
        const auto& b = e.get_ro_buf();
        std::fwrite(b.data(), sizeof(float), b.size(), o);
        std::fclose(o);
        std::printf("%s: %dx%d -> %dx%d\n", img.c_str(), u8.get_size().width, u8.get_size().height, e.nx(), e.ny());
        mtmd_bitmap_free(bw.bitmap);
    }
    clip_free(r.ctx_v);
    return rc;
}
