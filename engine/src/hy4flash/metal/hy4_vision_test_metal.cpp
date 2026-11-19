// hy4_vision_test_metal.cpp - ViT parity / timing program for the Metal vision path.
// usage: hy4-vision-test-metal --vision Vision.strata --image pic.png [--out DIR] [--kernels DIR/hy4.metal] [--reps N] [--probe]
//   writes emb0.npy, L0..L26.npy (fp32 residual stream after every block) and image_rows_gpu.npy into DIR (same names as the HIP test, compare with phaseA-tests/hy4/vision/cmp_vit.py)
#include <cstdio>
#include <string>

#include "hy4_metal_vision.h"

int main(int argc, char** argv) {
    std::string vision, image, out = ".", kernels = "hy4.metal";
    bool probe = false, dump = true;
    int reps = 1;
    int64_t max_px = 4194304;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nxt = [&]() { return std::string(argv[++i]); };
        if (a == "--vision") vision = nxt();
        else if (a == "--image") image = nxt();
        else if (a == "--out") out = nxt();
        else if (a == "--kernels") kernels = nxt();
        else if (a == "--reps") reps = atoi(nxt().c_str());
        else if (a == "--probe") probe = true;
        else if (a == "--nodump") dump = false;
        else if (a == "--max-pixels") max_px = atoll(nxt().c_str());
        else {
            fprintf(stderr, "unknown arg %s\n", a.c_str());
            return 2;
        }
    }
    mtl::Context cx;
    const std::string dir = kernels.substr(0, kernels.rfind('/') + 1);
    hy4::VitMetal vm;
    std::string err;
    if (!vm.load(cx, vision, dir, err)) {
        fprintf(stderr, "load: %s\n", err.c_str());
        return 1;
    }
    if (probe) {  // fragment layout of simdgroup_matrix: lane -> (row, col) of its two elements
        mtl::Buffer o = cx.alloc(64 * 4);
        mtl::Dispatch d;
        d.p = &vm.P("v_layout_probe");
        d.args = {mtl::Arg::B(o)};
        d.tg[0] = 32;
        cx.run({d});
        int bad = 0;
        for (int l = 0; l < 32; ++l) {
            const int qid = l / 4, fm = (qid & 4) + ((l / 2) % 4), fn = (qid & 2) * 2 + (l % 2) * 2;
            const float* f = (const float*) o.ptr;
            const int row = (int) f[l * 2] / 8, col = (int) f[l * 2] % 8;
            const int col1 = (int) f[l * 2 + 1] % 8;
            if (row != fm || col != fn || col1 != fn + 1) {
                ++bad;
                printf("lane %d: got (%d,%d,%d) expected (%d,%d,%d)\n", l, row, col, col1, fm, fn, fn + 1);
            }
        }
        printf("layout probe: %s\n", bad ? "MISMATCH" : "ok");
    }
    if (image.empty()) return 0;
    hy4::ImageRec rc;
    if (!hy4::prepare_picture(image, hy4::VitCfg().min_pixels, max_px, rc, err)) {
        fprintf(stderr, "image: %s\n", err.c_str());
        return 1;
    }
    const int rows = hy4::image_token_count(rc.h, rc.w);
    fprintf(stderr, "[image] %dx%d -> %d patches, %d LM rows\n", rc.w, rc.h, (rc.h / 16) * (rc.w / 16), rows);
    mtl::Buffer rb = cx.alloc((size_t) rows * 3072 * 4);
    vm.dump = dump;
    vm.dump_dir = out;
    vm.prof = true;
    for (int r = 0; r < reps; ++r) {
        if (!vm.encode(rc.rgb.data(), rc.h, rc.w, rb, 0, err)) {
            fprintf(stderr, "encode: %s\n", err.c_str());
            return 1;
        }
        fprintf(stderr, "[vit] run %d: wall %.3f s, gpu %.3f s\n", r, vm.last_wall, vm.t_gpu);
        vm.dump = false;
    }
    hy4::vis_save_npy_f32(out + "/image_rows_gpu.npy", (const float*) rb.ptr, {rows, 3072});
    return 0;
}
