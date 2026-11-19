// hy4_image_test.cpp - host-only check of the picture path (no GPU): decode + smart_resize + PIL-bicubic port.
//   hy4-image-test IN.png OUT.rgb  -> writes the resized RGB bytes and prints "W H"; compared with PIL by phaseA-tests/hy4/vision/cmp_resize.py
#include <cstdio>

#include "hy4_vision.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: hy4-image-test IN OUT.rgb [max_pixels]\n");
        return 2;
    }
    hy4::ImageRec rc;
    std::string err;
    const int64_t maxpx = argc > 3 ? atoll(argv[3]) : hy4::VitCfg().max_pixels;
    if (!hy4::prepare_picture(argv[1], hy4::VitCfg().min_pixels, maxpx, rc, err)) {
        fprintf(stderr, "image: %s\n", err.c_str());
        return 1;
    }
    FILE* f = fopen(argv[2], "wb");
    fwrite(rc.rgb.data(), 1, rc.rgb.size(), f);
    fclose(f);
    printf("%d %d\n", rc.w, rc.h);
    return 0;
}
