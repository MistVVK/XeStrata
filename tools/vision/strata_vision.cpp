// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// tools/vision/strata_vision.cpp - the chat's picture reader: the vision encoder the image engine shares
// (strata/core/vision.hpp: Qwen3-VL's ViT and its merger from the model's mmproj GGUF), on the GPU with --gpu and on
// the CPU without (int8 products by default, BF16 with XESTRATA_VISION_CPU_I8=0, on --threads threads, default one a
// core).  The protocol is the one llama.cpp's mtmd reader spoke before it, so serve and the engines do not change:
//
//   xestrata-vision --mmproj <mmproj.gguf> [--model <text model, for a config that names it: not read>]
//                   [--gpu] [--gpu-pci ADDR] [--threads N] [--max-tokens N] [--min-tokens N]
//
// Resident: prints "READY <n_embd>", then per stdin line
//   ENC <image path> <output path>   ->  "OK <n_tokens> <nx> <ny> <ms>"  or  "ERR <message>"
//   QUIT
// The output file is  int32 {0x31455653 'SVE1', n_tokens, nx, ny, n_embd}  then float32 [n_tokens][n_embd], row i at
// grid position (x = i % nx, y = i / nx): the format the engines' GENI reads.
//
// The picture becomes the tokens mtmd's qwen3vl reader makes of it (the same grid, so the prompt's pads and the
// answers stay as they were): the sides to a multiple of patch * merge (32), the aspect kept, between --min-tokens
// and --max-tokens, Pillow's bicubic, RGB in -1 .. 1 (the mmproj's image mean and deviation of 0.5).
#include "strata/core/runtime.hpp"
#include "strata/core/vision.hpp"
#include "strata/kernels/cpu/bf16_gemm.hpp"
#include "strata/kernels/cpu/i8_gemm.hpp"
#include "strata/image/img_crop.hpp"
#include "strata/image/img_start.hpp"
#include "strata/image/pipeline.hpp"
#include "strata/image/qi_image_io.hpp"
#include "strata/platform/env.hpp"
#include "strata/prefill/gemm.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Size {
    int64_t w = 0, h = 0;
};

// llama.cpp's mtmd_image_preprocessor_dyn_size (calc_size_preserved_ratio, tools/mtmd/mtmd-image.cpp): the sides
// rounded to `align` (the patch times the merge), then shrunk below max_pixels or grown above min_pixels with the
// aspect kept: the grid a qwen3vl reader makes of a picture.  The limits are tokens (the server's --min/--max).
Size reader_size(int64_t width, int64_t height, int64_t align, int64_t min_tokens, int64_t max_tokens) {
    const int64_t patch_area = align * align;
    const int64_t min_pixels = std::max<int64_t>(0, min_tokens) * patch_area;
    const int64_t max_pixels = std::max<int64_t>(0, max_tokens) * patch_area;
    auto round_by = [align](double x) { return (int64_t) std::llround(x / (double) align) * align; };
    auto ceil_by = [align](double x) { return (int64_t) std::ceil(x / (double) align) * align; };
    auto floor_by = [align](double x) { return (int64_t) std::floor(x / (double) align) * align; };
    int64_t w = std::max<int64_t>(align, round_by((double) width));
    int64_t h = std::max<int64_t>(align, round_by((double) height));
    if (max_pixels > 0 && h * w > max_pixels) {
        const double beta = std::sqrt((double) height * (double) width / (double) max_pixels);
        h = std::max<int64_t>(align, floor_by((double) height / beta));
        w = std::max<int64_t>(align, floor_by((double) width / beta));
    } else if (min_pixels > 0 && h * w < min_pixels) {
        const double beta = std::sqrt((double) min_pixels / ((double) height * (double) width));
        h = ceil_by((double) height * beta);
        w = ceil_by((double) width * beta);
    }
    return {w, h};
}

// The picture's planes [3][h][w] in -1 .. 1, the pixels mtmd's qwen3vl preprocessing makes of the same file: its
// img_tool::resize with PAD_CEIL (the hparams' default, and qwen3vl does not change it) scales the picture to fit the
// reader's size with its aspect kept - the sides to their ceilings, never past the target - and centres it on a canvas
// of the pad color (black).  The alpha is dropped: mtmd reads three channels.
std::vector<float> picture_planes(const std::vector<uint8_t>& rgba, int64_t width, int64_t height, const Size& size) {
    const float scale = std::min((float) size.w / (float) width, (float) size.h / (float) height);
    const int64_t new_w = std::min<int64_t>((int64_t) std::ceil((float) width * scale), size.w);
    const int64_t new_h = std::min<int64_t>((int64_t) std::ceil((float) height * scale), size.h);
    const strata::image::QiRef src{rgba, width, height};
    const strata::image::QiRef at =
        strata::image::pillow_resize(src, new_w, new_h, strata::image::Filter::bicubic);
    std::vector<uint8_t> canvas((size_t) (size.w * size.h * 4), 0);   // mtmd's pad color: black
    for (size_t i = 3; i < canvas.size(); i += 4) canvas[i] = 255;    // (the alpha plane is not read)
    const int64_t ox = (size.w - new_w) / 2, oy = (size.h - new_h) / 2;
    for (int64_t y = 0; y < new_h; ++y)
        std::memcpy(canvas.data() + (size_t) ((y + oy) * size.w + ox) * 4,
                    at.rgba.data() + (size_t) (y * new_w) * 4, (size_t) new_w * 4);
    return strata::image::image_planes(canvas, size.w, size.h, 3, strata::image::PlaneAlpha::Keep,
                                       strata::image::PlaneRange::Signed);
}

// "ENC <image> <out>": paths may contain spaces, so the image path ends at the last " " before the output path (as
// mtmd's reader read it; the server writes paths without spaces)
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

int main(int argc, char** argv) {
    std::string mmproj, model, gpu_pci;
    int64_t max_tokens = 0, min_tokens = 0;
    bool gpu = false;
    int threads = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--mmproj") mmproj = next();
        else if (a == "--model") model = next();   // the configs name it (mtmd's reader read its vocabulary); not read
        else if (a == "--gpu") gpu = true;
        else if (a == "--gpu-pci") gpu_pci = next();
        else if (a == "--threads") threads = (int) std::strtol(next().c_str(), nullptr, 10);
        else if (a == "--max-tokens") max_tokens = std::strtoll(next().c_str(), nullptr, 10);
        else if (a == "--min-tokens") min_tokens = std::strtoll(next().c_str(), nullptr, 10);
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (mmproj.empty()) {
        std::fprintf(stderr, "usage: xestrata-vision --mmproj <mmproj.gguf> [--model <model.gguf>] [--gpu] "
                             "[--gpu-pci ADDR] [--threads N] [--max-tokens N] [--min-tokens N]\n");
        return 2;
    }
    (void) model;
    // on the CPU without --threads: one per core (hardware threads over two, as mtmd's reader took them)
    if (threads <= 0) threads = (int) std::max(1u, std::thread::hardware_concurrency() / 2);
    // the ViT's attention in 16 bits unless the environment names a choice: the rows come out closer to the FP32
    // reference than the int8 kernel's, for about 40 ms a picture of 1,024 tokens
    if (gpu && strata::env_get("IMAGE_ATTN_I8") == nullptr) {
#if defined(_WIN32)
        _putenv_s("XESTRATA_IMAGE_ATTN_I8", "0");
#else
        setenv("XESTRATA_IMAGE_ATTN_I8", "0", 0);
#endif
    }
    // the runtime reads the card's PCI address when it starts (core/device.cpp: XESTRATA_GPU_PCI), as the engines take
    // it: this has to happen before the first device is opened
    if (gpu && !gpu_pci.empty()) {
#if defined(_WIN32)
        _putenv_s("XESTRATA_GPU_PCI", gpu_pci.c_str());
#else
        setenv("XESTRATA_GPU_PCI", gpu_pci.c_str(), 1);
#endif
    }

    // the products' library loaded on a thread of its own while the encoder opens (as the image engine starts), and not
    // still loading when the process ends (the GPU's; the CPU opens no device)
    struct GemmDone {
        bool on = false;
        ~GemmDone() {
            if (on) strata::prefill::Gemm::finish();
        }
    } gemm_done;
    void* stream = nullptr;
    if (gpu) {
        stream = strata::core::Runtime::get().create_stream();
        strata::prefill::Gemm::prepare();
        strata::prefill::Gemm::settle();
        gemm_done.on = true;
    }
    strata::core::Vision vision;
    std::string err;
    const auto load_t0 = std::chrono::steady_clock::now();
    if (!(gpu ? vision.open(mmproj, stream, err) : vision.open_cpu(mmproj, threads, err))) {
        std::printf("ERR cannot load the vision encoder %s (%s)\n", mmproj.c_str(), err.c_str());
        std::fflush(stdout);
        return 1;
    }
    std::fprintf(stderr, "xestrata-vision: model files loaded in %.1f s (weights %.2f GB)\n",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - load_t0).count(),
                 (double) vision.weight_bytes() / 1e9);
    if (!gpu)
        std::fprintf(stderr, "xestrata-vision: on the CPU, %d threads, %s products (%s), no warm-up\n", threads,
                     vision.cpu_i8() ? "int8" : "BF16",
                     vision.cpu_i8() ? strata::kernels::cpu::i8_gemm_isa_name(strata::kernels::cpu::i8_gemm_isa())
                                     : strata::kernels::cpu::bf16_gemm_isa_name(strata::kernels::cpu::bf16_gemm_isa()));
    const int64_t align = vision.patch() * vision.merge();
    // Warm up at the largest picture (well above any cap; the reader's size rule brings it to --max-tokens) before
    // READY: the encoder's work buffers are allocated now, not at the first real picture, and the server starts this
    // process before the engine so that the engine sizes its expert cache from what is really left.  The CPU's
    // buffers are the host's, so it only counts the tokens such a picture becomes (an encode there takes seconds).
    int64_t warm_tokens = 0, n_embd = 0;
    if (!gpu) {
        const Size sz = reader_size(2048, 2048, align, min_tokens, max_tokens);
        warm_tokens = (sz.w / align) * (sz.h / align);
        n_embd = vision.dim();
        if (max_tokens > 0 && warm_tokens > max_tokens + max_tokens / 8)
            std::fprintf(stderr, "xestrata-vision: WARNING a 2048 x 2048 picture becomes %lld image tokens, over the "
                                 "limit of %lld (--max-tokens): pictures over it are refused\n",
                         (long long) warm_tokens, (long long) max_tokens);
    } else {
        const auto warm_t0 = std::chrono::steady_clock::now();
        const int64_t side = 2048;
        std::vector<uint8_t> gray((size_t) (side * side * 4), 128);
        const Size sz = reader_size(side, side, align, min_tokens, max_tokens);
        try {
            const auto rgb = picture_planes(gray, side, side, sz);
            const strata::core::VisionOut out = vision.encode(rgb.data(), sz.w, sz.h);
            warm_tokens = out.rows;
            n_embd = out.dim;
        } catch (const std::exception& e) {
            std::printf("ERR the warm-up encode failed: %s\n", e.what());
            std::fflush(stdout);
            return 1;
        }
        std::fprintf(stderr, "xestrata-vision: warmed up at %lld image tokens in %.1f s\n", (long long) warm_tokens,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - warm_t0).count());
        // I7 (#3690): a picture that became 93,700 image tokens under --max-tokens 1024: the evidence line the
        // report asks for
        if (max_tokens > 0 && warm_tokens > max_tokens + max_tokens / 8)
            std::fprintf(stderr, "xestrata-vision: WARNING the warm-up picture became %lld image tokens, over the limit "
                                 "of %lld (--max-tokens): pictures over it are refused\n", (long long) warm_tokens,
                         (long long) max_tokens);
    }
    if (warm_tokens <= 0 || n_embd <= 0) {
        std::printf("ERR the warm-up encode failed\n");
        std::fflush(stdout);
        return 1;
    }
    std::printf("READY %lld\n", (long long) n_embd);
    std::fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line == "QUIT") break;
        std::string img, out_path;
        if (!parse_enc(line, img, out_path)) {
            std::printf("ERR expected: ENC <image> <output>\n");
            std::fflush(stdout);
            continue;
        }
        const auto t0 = std::chrono::steady_clock::now();
        int64_t w = 0, h = 0;
        const std::vector<uint8_t> rgba = strata::image::load_rgba8(img, w, h, err);
        if (rgba.empty() || w <= 0 || h <= 0) {
            std::printf("ERR cannot read the image %s (%s)\n", img.c_str(), err.c_str());
            std::fflush(stdout);
            continue;
        }
        const Size sz = reader_size(w, h, align, min_tokens, max_tokens);
        strata::core::VisionOut out;
        try {
            const std::vector<float> rgb = picture_planes(rgba, w, h, sz);
            out = vision.encode(rgb.data(), sz.w, sz.h);
        } catch (const std::exception& e) {
            std::printf("ERR %s\n", e.what());
            std::fflush(stdout);
            continue;
        }
        const int64_t n = out.rows, nx = out.grid_w, ny = out.grid_h;
        if (nx * ny != n || out.dim != n_embd) {
            std::printf("ERR the image grid is not rectangular (%lld tokens)\n", (long long) n);
            std::fflush(stdout);
            continue;
        }
        // never silently produce many times the budget (I7, #3690): over the limit (a rounding margin of an eighth)
        // the picture is refused, as mtmd's reader refused it
        if (max_tokens > 0 && n > max_tokens + max_tokens / 8) {
            std::printf("ERR the picture became %lld image tokens, over the limit of %lld (vision.max_tokens); send a "
                        "smaller picture or raise the limit\n", (long long) n, (long long) max_tokens);
            std::fflush(stdout);
            continue;
        }
        FILE* f = std::fopen(out_path.c_str(), "wb");
        if (f == nullptr) {
            std::printf("ERR cannot write %s\n", out_path.c_str());
            std::fflush(stdout);
            continue;
        }
        const int32_t hdr[5] = {0x31455653, (int32_t) n, (int32_t) nx, (int32_t) ny, (int32_t) out.dim};
        const bool ok = std::fwrite(hdr, sizeof hdr, 1, f) == 1 &&
                        std::fwrite(out.main.data(), sizeof(float) * (size_t) out.dim, (size_t) n, f) == (size_t) n;
        std::fclose(f);
        if (!ok) {
            std::printf("ERR short write to %s\n", out_path.c_str());
            std::fflush(stdout);
            continue;
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("OK %lld %lld %lld %.0f\n", (long long) n, (long long) nx, (long long) ny, ms);
        std::fflush(stdout);
    }
    return 0;
}
