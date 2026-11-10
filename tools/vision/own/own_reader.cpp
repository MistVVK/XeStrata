// tools/vision/own/own_reader.cpp - see own_reader.h
#include "own_reader.h"

#include "cpu_caps.h"
#include "vit_cpu.h"
#include "vit_ref.h"
#ifdef STRATA_VIT_GPU
#include "vit_gpu.h"
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
        if (!VitGpu::available(why) || VitGpu::prefer_mtmd(why)) return false;
#else
        why = "this build has no GPU backend for the own tower";
        return false;
#endif
    }
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    else if (quant::best_isa() == quant::Isa::Scalar) {
        // an x86 CPU with only the portable kernel tier (no AVX2): measured 2.1 to 3.7 times mtmd-CPU's time (RESULTS-OWN-3V.md 12.7), so `auto` keeps mtmd
        // here; `--tower own` still runs ours.  Not on Arm: mtmd's portable Arm build has no fast BF16 path and our NEON tier is faster than mtmd-CPU on BF16
        // and F16 weights alike (DGX Spark, 12.12)
        why = "only the portable CPU kernels on this CPU: the own CPU tower is slower than mtmd there";
        return false;
    }
#endif
    else if (vit_cpu_build_slower_than_mtmd()) {
        why = "this build of the own CPU tower (MSVC, x86) is slower than mtmd's CPU reader";
        return false;
    }
    Gguf gg;
    VitGeometry g;
    if (!gg.open(mmproj, why) || !vit_geometry_from_gguf(gg, g, why)) return false;
    if (!vit_geometry_supported(g, why)) return false;
    if (gpu && g.head_dim > 96) { why = "a head size above 96 is not handled by the GPU attention"; return false; }   // `auto` then reads with mtmd
    return true;
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
    std::unique_ptr<VitCpu> fast;
#ifdef STRATA_VIT_GPU
    std::unique_ptr<VitGpu> gpu;
    if (o.gpu) {
        std::string why;
        if (!VitGpu::available(why)) return fail("this picture reader found no GPU (" + why + ")", 3);
        gpu.reset(new VitGpu());
        // buffers for the warm-up picture below (2048 x 2048: 4,096 tokens at most, fewer under a lower limit): claimed now, before READY.  A larger limit
        // (the app allows up to 16,384 image tokens) grows them when such a picture comes: slower for that picture, never an error.
        const int tokens = std::min(o.max_tokens > 0 ? o.max_tokens : geo.max_tokens, 4096);
        if (!gpu->load(gg, geo, 0, tokens * 4, err)) return fail("cannot load the vision encoder " + o.mmproj + " (" + err + ")", 1);
    }
#endif
    if (!o.gpu) {
        // the CPU path of the own tower is the fast tower (VitCpu); STRATA_VIT_CPU=ref selects the float32 reference (VitRef: the oracle, for tests)
        const char* sel = std::getenv("STRATA_VIT_CPU");
        if (!(sel && std::string(sel) == "ref")) {
            fast.reset(new VitCpu());
            if (!fast->load(gg, geo, err)) return fail("cannot load the vision encoder " + o.mmproj + " (" + err + ")", 1);
        } else {
            cpu.reset(new VitRef());
            if (!cpu->load(gg, geo, err)) return fail("cannot load the vision encoder " + o.mmproj + " (" + err + ")", 1);
        }
        if (threads <= 0) threads = (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency() / 2));
    }
    std::fprintf(stderr, "strata-vision: own tower (%s, %d layers, width %d) loaded in %.1f s, %s\n", geo.projector.c_str(), geo.layers, geo.width,
                 std::chrono::duration<double>(clock::now() - load_t0).count(), o.gpu ? "on the GPU" : ("on the CPU, " + std::to_string(threads) + " threads").c_str());
    auto encode = [&](const PreparedImage& prep, std::vector<float>& rows, int& nx, int& ny, std::string& e) {
#ifdef STRATA_VIT_GPU
        if (gpu) return gpu->encode(prep, rows, nx, ny, e);
#endif
        if (fast) return fast->encode(prep, threads, rows, nx, ny, e);
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
        if (std::getenv("STRATA_VIT_PROF"))
            std::fprintf(stderr, "vit-prof: read + decode + prepare %.2f ms\n", std::chrono::duration<double, std::milli>(clock::now() - t0).count());
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
