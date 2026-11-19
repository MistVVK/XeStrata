// hy4_vision_test.hip.cpp - strata-hy4-vision-test: runs the GPU ViT + adapter on one resized RGB picture and writes the image rows (and with --dump every layer) for the parity check
// against hy4_vit_ref.py (phaseA-tests/hy4/vision).
//   strata-hy4-vision-test --vision Hy4-Flash-Vision.strata --rgb rgb.u8 --h H --w W --out DIR [--dump] [--reps N]
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "hy4_vision.hip.h"

using namespace hy4;

int main(int argc, char** argv) {
    std::string shard, rgbp, imgp, out = ".";
    int h = 0, w = 0, reps = 1;
    bool dump = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nxt = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--vision") shard = nxt();
        else if (a == "--rgb") rgbp = nxt();
        else if (a == "--image") imgp = nxt();
        else if (a == "--h") h = atoi(nxt().c_str());
        else if (a == "--w") w = atoi(nxt().c_str());
        else if (a == "--out") out = nxt();
        else if (a == "--dump") dump = true;
        else if (a == "--reps") reps = atoi(nxt().c_str());
        else {
            fprintf(stderr, "unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    if (shard.empty() || (imgp.empty() && (rgbp.empty() || h <= 0 || w <= 0))) {
        fprintf(stderr, "usage: strata-hy4-vision-test --vision SHARD (--rgb rgb.u8 --h H --w W | --image FILE) --out DIR [--dump] [--reps N]\n");
        return 2;
    }
    std::vector<uint8_t> rgb;
    if (!imgp.empty()) {  // decode + resize in the engine (stb + the PIL bicubic port); the pixels are kept for the comparison with PIL
        ImageRec rc;
        std::string ierr;
        if (!prepare_picture(imgp, VitCfg().min_pixels, VitCfg().max_pixels, rc, ierr)) {
            fprintf(stderr, "image: %s\n", ierr.c_str());
            return 2;
        }
        rgb = rc.rgb;
        w = rc.w;
        h = rc.h;
        FILE* fo = fopen((out + "/rgb_cpp.u8").c_str(), "wb");
        if (fo) {
            fwrite(rgb.data(), 1, rgb.size(), fo);
            fclose(fo);
        }
        fprintf(stderr, "[image] %s -> %dx%d\n", imgp.c_str(), w, h);
    } else {
        rgb.resize((size_t)h * w * 3);
        FILE* f = fopen(rgbp.c_str(), "rb");
        if (!f || fread(rgb.data(), 1, rgb.size(), f) != rgb.size()) {
            fprintf(stderr, "cannot read %s (%zu bytes)\n", rgbp.c_str(), rgb.size());
            return 2;
        }
        fclose(f);
    }
    hipDeviceProp_t p;
    HCK(hipGetDeviceProperties(&p, 0));
    fprintf(stderr, "[gpu] %s %s\n", p.name, gpu_arch(p));
    hipStream_t st;
    HCK(hipStreamCreate(&st));
    VisionEncoder enc;
    enc.dump = dump;
    enc.dump_dir = out;
    std::string err;
    if (!enc.load(shard, err)) {
        fprintf(stderr, "load: %s\n", err.c_str());
        return 1;
    }
    const int rows = image_token_count(h, w);
    float* d_rows = nullptr;
    HCK(hipMalloc((void**)&d_rows, (size_t)rows * enc.vi.c.out_hidden * 4));
    for (int r = 0; r < reps; ++r) {
        double t0 = now_s();
        if (!enc.encode(st, rgb.data(), h, w, d_rows, err)) {
            fprintf(stderr, "encode: %s\n", err.c_str());
            return 1;
        }
        fprintf(stderr, "[vision] %dx%d: %d patches, %d rows, encode %.3f s%s\n", w, h, (h / 16) * (w / 16), rows, now_s() - t0, r == 0 ? " (first run)" : "");
        enc.dump = false;
    }
    std::vector<float> hr((size_t)rows * enc.vi.c.out_hidden);
    HCK(hipMemcpy(hr.data(), d_rows, hr.size() * 4, hipMemcpyDeviceToHost));
    VisionEncoder::save_npy_f32_vis(out + "/image_rows_gpu.npy", hr.data(), {rows, enc.vi.c.out_hidden});
    return 0;
}
