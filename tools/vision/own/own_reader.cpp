// tools/vision/own/own_reader.cpp - see own_reader.h
#include "own_reader.h"

#include "vit_ref.h"
#ifdef STRATA_VIT_GPU
#include "vit_gpu.h"
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <thread>

namespace strata_vit {

namespace {

// "ENC <image> <out>": the image path ends at the last space before the output path (a path of the image may hold spaces)
bool parse_enc(const std::string& line, std::string& img, std::string& out) {
    if (line.rfind("ENC ", 0) != 0) return false;
    const std::string rest = line.substr(4);
    const size_t sp = rest.rfind(' ');
    if (sp == std::string::npos || sp == 0) return false;
    img = rest.substr(0, sp);
    out = rest.substr(sp + 1);
    return !img.empty() && !out.empty();
}

}  // namespace

bool own_reader_supported(const std::string& mmproj, bool gpu, std::string& why) {
    if (gpu) {
#ifdef STRATA_VIT_GPU
        if (!VitGpu::available(why)) return false;
#else
        why = "this build has no GPU backend for the own tower";
        return false;
#endif
    }
    Gguf gg;
    VitGeometry g;
    if (!gg.open(mmproj, why) || !vit_geometry_from_gguf(gg, g, why)) return false;
    return vit_geometry_supported(g, why);
}

int own_reader_run(const OwnReaderOptions& o) {
    using clock = std::chrono::steady_clock;
    const auto load_t0 = clock::now();
    auto fail = [](const std::string& msg, int code) {
        std::printf("ERR %s\n", msg.c_str());
        std::fflush(stdout);
        return code;
    };
#ifndef STRATA_VIT_GPU
    if (o.gpu) return fail("this build has no GPU backend for the own tower (use --tower mtmd)", 3);
#endif
    Gguf gg;
    VitGeometry geo;
    std::string err;
    if (!gg.open(o.mmproj, err) || !vit_geometry_from_gguf(gg, geo, err) || !vit_geometry_supported(geo, err))
        return fail("cannot load the vision encoder " + o.mmproj + " (" + err + ")", 1);
    int threads = o.threads;
    std::unique_ptr<VitRef> cpu;
#ifdef STRATA_VIT_GPU
    std::unique_ptr<VitGpu> gpu;
    if (o.gpu) {
        std::string why;
        if (!VitGpu::available(why)) return fail("this picture reader found no GPU (" + why + ")", 3);
        gpu.reset(new VitGpu());
        // buffers for the largest picture the limit allows (without a limit: the tower's own 4,096 tokens): claimed now, before READY
        const int tokens = o.max_tokens > 0 ? o.max_tokens + o.max_tokens / 8 : geo.max_tokens;
        if (!gpu->load(gg, geo, 0, tokens * 4, err)) return fail("cannot load the vision encoder " + o.mmproj + " (" + err + ")", 1);
    }
#endif
    if (!o.gpu) {
        cpu.reset(new VitRef());
        if (!cpu->load(gg, geo, err)) return fail("cannot load the vision encoder " + o.mmproj + " (" + err + ")", 1);
        if (threads <= 0) threads = (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency() / 2));
    }
    std::fprintf(stderr, "strata-vision: own tower (%s, %d layers, width %d) loaded in %.1f s, %s\n", geo.projector.c_str(), geo.layers, geo.width,
                 std::chrono::duration<double>(clock::now() - load_t0).count(), o.gpu ? "on the GPU" : ("on the CPU, " + std::to_string(threads) + " threads").c_str());
    auto encode = [&](const PreparedImage& prep, std::vector<float>& rows, int& nx, int& ny, std::string& e) {
#ifdef STRATA_VIT_GPU
        if (gpu) return gpu->encode(prep, rows, nx, ny, e);
#endif
        return cpu->encode(prep, threads, rows, nx, ny, e);
    };
#ifdef STRATA_VIT_GPU
    if (gpu) {   // warm up at the largest picture before READY: the work buffers are taken now, not at the first real picture
        const auto w0 = clock::now();
        ImageU8 gray;
        gray.w = gray.h = 2048;
        gray.rgb.assign((size_t) 2048 * 2048 * 3, 128);
        PreparedImage wp;
        std::vector<float> rows;
        int nx = 0, ny = 0;
        if (prepare_image(gray, geo, o.min_tokens, o.max_tokens, wp, err) && encode(wp, rows, nx, ny, err))
            std::fprintf(stderr, "strata-vision: warmed up at %d image tokens in %.1f s, %.0f MiB of GPU memory\n", nx * ny,
                         std::chrono::duration<double>(clock::now() - w0).count(), (double) gpu->device_bytes() / 1048576.0);
        else
            return fail("the vision encoder failed its warm-up (" + err + ")", 1);
    }
#endif
    std::printf("READY %d\n", geo.out_dim);
    std::fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line == "QUIT") break;
        std::string img, out;
        if (!parse_enc(line, img, out)) { std::printf("ERR expected: ENC <image> <output>\n"); std::fflush(stdout); continue; }
        const auto t0 = clock::now();
        ImageU8 src;
        PreparedImage prep;
        if (!image_load_file(img, src, err)) { std::printf("ERR cannot read the image %s\n", img.c_str()); std::fflush(stdout); continue; }
        if (!prepare_image(src, geo, o.min_tokens, o.max_tokens, prep, err)) { std::printf("ERR the image could not be preprocessed (%s)\n", err.c_str()); std::fflush(stdout); continue; }
        const int n_expect = (prep.w / geo.align()) * (prep.h / geo.align());
        // never silently produce many times the budget (I7, #3690): over the limit (a rounding margin of an eighth) the picture is refused
        if (o.max_tokens > 0 && n_expect > o.max_tokens + o.max_tokens / 8) {
            std::printf("ERR the picture became %d image tokens, over the limit of %d (vision.max_tokens); send a smaller picture or raise the limit\n", n_expect,
                        o.max_tokens);
            std::fflush(stdout);
            continue;
        }
        std::vector<float> rows;
        int nx = 0, ny = 0;
        if (!encode(prep, rows, nx, ny, err)) { std::printf("ERR the vision encoder failed (%s)\n", err.c_str()); std::fflush(stdout); continue; }
        const int n = nx * ny;
        FILE* f = std::fopen(out.c_str(), "wb");
        if (!f) { std::printf("ERR cannot write %s\n", out.c_str()); std::fflush(stdout); continue; }
        const int32_t hdr[5] = {0x31455653, n, nx, ny, geo.out_dim};
        const bool ok = std::fwrite(hdr, sizeof hdr, 1, f) == 1 && std::fwrite(rows.data(), sizeof(float) * (size_t) geo.out_dim, (size_t) n, f) == (size_t) n;
        std::fclose(f);
        if (!ok) { std::printf("ERR short write to %s\n", out.c_str()); std::fflush(stdout); continue; }
        const double ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
        std::printf("OK %d %d %d %.0f\n", n, nx, ny, ms);
        std::fflush(stdout);
    }
    return 0;
}

}  // namespace strata_vit
