// sy_vit_test.cpp - runs the SYCL ViT + adapter on one resized RGB picture and dumps the residual stream per layer and the image rows (compare with phaseA-tests/hy4/vision/cmp_vit.py).
// usage: sy_vit_test <Hy4-Flash-Vision.strata> <pic.rgb (raw u8 h*w*3)> <h> <w> <outdir> [reps]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "sy_vision.h"

using namespace hy4s;

static void save_rows(const std::string& path, const float* d, int R, int D) {
    std::string sh = "(" + std::to_string(R) + "," + std::to_string(D) + ")";
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': " + sh + ", }";
    const size_t tot = 10 + hdr.size() + 1, pad = (64 - tot % 64) % 64;
    hdr += std::string(pad, ' ') + "\n";
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    fwrite("\x93NUMPY", 1, 6, f);
    const unsigned char ver[2] = {1, 0};
    fwrite(ver, 1, 2, f);
    const uint16_t hl = (uint16_t)hdr.size();
    fwrite(&hl, 2, 1, f);
    fwrite(hdr.data(), 1, hdr.size(), f);
    fwrite(d, 4, (size_t)R * D, f);
    fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: sy_vit_test shard.strata pic.rgb h w outdir [reps]\n");
        return 2;
    }
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) hy4s_setenv("ONEAPI_DEVICE_SELECTOR", "opencl:gpu");
    const std::string shard = argv[1], rgbf = argv[2], outdir = argv[5];
    const int h = atoi(argv[3]), w = atoi(argv[4]);
    const int reps = argc > 6 ? atoi(argv[6]) : 1;
    std::vector<uint8_t> rgb((size_t)h * w * 3);
    {
        std::ifstream f(rgbf, std::ios::binary);
        f.read((char*)rgb.data(), (std::streamsize)rgb.size());
        if (!f) {
            fprintf(stderr, "cannot read %s\n", rgbf.c_str());
            return 1;
        }
    }
    try {
        printf("device %s\n", Q().get_device().get_info<sycl::info::device::name>().c_str());
        VisionEncoder enc;
        std::string err;
        if (!enc.load(shard, err)) {
            fprintf(stderr, "load: %s\n", err.c_str());
            return 1;
        }
        const int oh = h / 32, ow = w / 32, R = oh * (ow + 1);
        float* rows = dalloc<float>((size_t)R * 3072);
        enc.dump_dir = outdir;
        if (!enc.encode(rgb.data(), h, w, rows, err)) {
            fprintf(stderr, "encode: %s\n", err.c_str());
            return 1;
        }
        std::vector<float> hr((size_t)R * 3072);
        d2h(hr.data(), rows, hr.size() * 4);
        wait();
        save_rows(outdir + "/image_rows_gpu.npy", hr.data(), R, 3072);
        enc.dump_dir.clear();
        for (int i = 0; i < reps; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            if (!enc.encode(rgb.data(), h, w, rows, err)) return 1;
            wait();
            printf("encode %d: %.3f s (%d patches, %d rows)\n", i, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), (h / 16) * (w / 16), R);
        }
    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL exception: %s\n", e.what());
        return 4;
    }
    return 0;
}
