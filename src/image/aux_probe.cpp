// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/aux_probe.cpp - strata-image-aux-probe: the image tools' auxiliary models' parts against their references
// (a record's, dumped by hand in a throwaway venv), and qi_sdpa_ex's shapes against FP64.
//
//   strata-image-aux-probe sdpa
//       qi_sdpa_ex's own kernels (rows and few) and its library path against an FP64 attention: independent query and
//       key lengths, batches sharing keys, causal rows, padded keys, a relative position bias and shifted windows'
//       masks; exits 1 on a difference past the tolerance.
//   strata-image-aux-probe sam3 <weights.safetensors> <image> <ids,...> <outdir> [score] [mask]
//       SAM 3 on the picture and the words' ids: its parts as raw FP32 files in outdir (embed, vit<i>, fpn<i>, text,
//       encoder, queries, scores, boxes, mask_logits), each instance's mask as mask<i>.u8 (raw, 255 inside), the
//       boxes, scores and times (a cold run, then a warm one) on stdout.
//   strata-image-aux-probe birefnet <weights.safetensors> <image> <outdir> [width [height]]
//       BiRefNet on the picture at width x height (1024; height: the width): its parts as raw FP32 files in outdir (x1 .. x4, squeeze,
//       p4 .. p1, logits), the alpha as alpha.u8 (raw, the picture's size), the foreground's colors (refine) as
//       fg.rgb (raw RGB8), the times (a cold run, then warm ones).
//   strata-image-aux-probe florence <model.safetensors> <image> <ids,...> <outdir> [beams] [max_new]
//       Florence-2 on the picture and the task's ids: its parts as raw FP32 files in outdir (vision_out, stage<i>,
//       image_features, encoder, logits0), the tokens and the times on stdout.
//   strata-image-aux-probe mask <W> <H> <outfile> <x,y,x,y,...>...
//       The polygons' mask (aux_polygon_fill), raw bytes (255 inside).

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/birefnet.hpp"
#include "strata/image/florence2.hpp"
#include "strata/image/sam3.hpp"
#include "strata/image/qi_image_io.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/qi_sdpa_ex.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/platform/env.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace si = strata::image;
namespace gpu = strata::gpu;
using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;

namespace {

std::vector<int32_t> ints(const std::string& s) {
    std::vector<int32_t> v;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, ',');)
        if (!t.empty()) v.push_back((int32_t) std::stol(t));
    return v;
}

std::vector<float> floats(const std::string& s) {
    std::vector<float> v;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, ',');)
        if (!t.empty()) v.push_back(std::stof(t));
    return v;
}

template <class T> T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    if (!gpu::alloc_device(&p, std::max<size_t>(h.size(), 1) * sizeof(T))) throw std::runtime_error("GPU memory");
    if (!h.empty()) gpu::copy(p, h.data(), h.size() * sizeof(T));
    return p;
}

// one shape: random q, k, v; the FP64 attention against qi_sdpa_ex's (and its library path where it takes the shape)
bool sdpa_case(const char* name, si::QiSdpa a, int64_t kv_batches, bool use_klen, bool use_bias, bool use_mask,
               std::mt19937& rng) {
    const int64_t row = a.heads * a.hd;
    a.q_row = a.k_row = a.v_row = a.o_row = row;
    a.q_batch = a.o_batch = a.lq * row;
    a.k_batch = a.v_batch = a.lk * row;
    std::normal_distribution<float> nd(0.f, 1.f);
    auto rnd16 = [&](int64_t n) {
        std::vector<uint16_t> v((size_t) n);
        for (auto& x : v) x = f16_from_f32(nd(rng));
        return v;
    };
    const auto q = rnd16(a.batch * a.lq * row), k = rnd16(kv_batches * a.lk * row), v = rnd16(kv_batches * a.lk * row);
    std::vector<int32_t> klen((size_t) a.batch);
    for (int64_t b = 0; b < a.batch; ++b) klen[(size_t) b] = (int32_t) std::max<int64_t>(1, a.lk - 3 * b);
    std::vector<float> bias((size_t) (a.heads * a.lq * a.lk)), mask((size_t) (a.mask_n * a.lq * a.lk));
    for (auto& x : bias) x = nd(rng);
    for (size_t i = 0; i < mask.size(); ++i) mask[i] = (i * 7 + i / 5) % 3 == 0 ? -100.f : 0.f;
    std::vector<uint16_t> out((size_t) (a.batch * a.lq * row));
    a.q = dev(q);
    a.k = dev(k);
    a.v = dev(v);
    a.out = dev(out);
    a.key_len = use_klen ? dev(klen) : nullptr;
    a.bias = use_bias ? dev(bias) : nullptr;
    a.mask = use_mask ? dev(mask) : nullptr;
    a.scale = 1.f / std::sqrt((float) a.hd);
    double worst = 0.0;
    bool ok = true;
    for (int path = 0; path < 3; ++path) {
        if (path == 0) {
            si::qi_sdpa_ex(a, nullptr);
        } else if (path == 1) {
            if (!si::qi_sdpa_ex_library(a, nullptr)) continue;
        } else {
            si::qi_sdpa_ex_own(a, nullptr);   // the own kernel forced (the matrix paths, not the library's choice)
        }
        gpu::device_sync();
        gpu::copy(out.data(), a.out, out.size() * 2);
        for (int64_t b = 0; b < a.batch; ++b)
            for (int64_t h = 0; h < a.heads; ++h)
                for (int64_t i = 0; i < a.lq; ++i) {
                    const int64_t kb = b / a.kv_div;
                    int64_t nk = use_klen ? klen[(size_t) b] : a.lk;
                    if (a.causal) nk = std::min(nk, i + a.causal_off + 1);
                    std::vector<double> s((size_t) std::max<int64_t>(nk, 0));
                    double m = -1e300;
                    for (int64_t j = 0; j < nk; ++j) {
                        double d = 0.0;
                        for (int64_t c = 0; c < a.hd; ++c)
                            d += (double) f32_from_f16(q[(size_t) ((b * a.lq + i) * row + h * a.hd + c)]) *
                                 f32_from_f16(k[(size_t) ((kb * a.lk + j) * row + h * a.hd + c)]);
                        d *= a.scale;
                        if (use_bias) d += bias[(size_t) ((h * a.lq + i) * a.lk + j)];
                        if (use_mask) d += mask[(size_t) (((b % a.mask_n) * a.lq + i) * a.lk + j)];
                        s[(size_t) j] = d;
                        m = std::max(m, d);
                    }
                    double z = 0.0;
                    for (double& x : s) z += (x = std::exp(x - m));
                    for (int64_t c = 0; c < a.hd; ++c) {
                        double o = 0.0;
                        for (int64_t j = 0; j < nk; ++j)
                            o += s[(size_t) j] * f32_from_f16(v[(size_t) ((kb * a.lk + j) * row + h * a.hd + c)]);
                        o = z > 0 ? o / z : 0.0;
                        const double got = f32_from_f16(out[(size_t) ((b * a.lq + i) * row + h * a.hd + c)]);
                        worst = std::max(worst, std::abs(got - o));
                    }
                }
        // FP16 out of values about 1: its half ulp (5e-4) and the scores' rounding
        const bool pass = worst < 4e-3;
        std::printf("sdpa %-34s %s max|err| %.2e%s\n", name, path == 0 ? "own/any" : path == 1 ? "library" : "own/forced",
                    worst, pass ? "" : "  FAIL");
        ok = ok && pass;
        worst = 0.0;
    }
    for (const void* p : {(const void*) a.q, (const void*) a.k, (const void*) a.v, (const void*) a.out,
                          (const void*) a.key_len, (const void*) a.bias, (const void*) a.mask})
        if (p != nullptr) gpu::free(p);
    return ok;
}

int sdpa_selftest() {
    std::mt19937 rng(7);   // NOLINT(bugprone-random-generator-seed): the same cases every run
    bool ok = true;
    si::QiSdpa a;
    // the encoders: one batch, no mask (the library's shape)
    a.batch = 1, a.heads = 12, a.lq = a.lk = 97, a.hd = 64;
    ok = sdpa_case("encoder 97x97 h12 d64", a, 1, false, false, false, rng) && ok;
    // a large unmasked encoder (the shared flash_attn, the head dim padded to 128) and a large biased shape
    a = {}, a.batch = 1, a.heads = 8, a.lq = a.lk = 1500, a.hd = 64;
    ok = sdpa_case("encoder 1500x1500 h8 d64", a, 1, false, false, false, rng) && ok;
    a = {}, a.batch = 2, a.heads = 4, a.lq = 300, a.lk = 300, a.hd = 64, a.mask_n = 2;
    ok = sdpa_case("biased 2x300 h4 d64", a, 2, false, true, true, rng) && ok;
    // DaViT's windows: 9 batches of 144, heads of 32
    a = {}, a.batch = 9, a.heads = 4, a.lq = a.lk = 144, a.hd = 32;
    ok = sdpa_case("windows 9x144 h4 d32", a, 9, false, false, false, rng) && ok;
    // a decoder step: 3 beams of one row, their own keys (the cache), padded
    a = {}, a.batch = 3, a.heads = 12, a.lq = 1, a.lk = 40, a.hd = 64;
    ok = sdpa_case("decoder self 3x1/40 padded", a, 3, true, false, false, rng) && ok;
    // the cross-attention: 3 beams over the encoder's shared keys
    a = {}, a.batch = 3, a.kv_div = 3, a.heads = 12, a.lq = 1, a.lk = 300, a.hd = 64;
    ok = sdpa_case("decoder cross 3x1/300 shared", a, 1, false, false, false, rng) && ok;
    // causal rows (a decoder's prompt), queries past the keys' start
    a = {}, a.batch = 2, a.heads = 2, a.lq = 33, a.lk = 50, a.hd = 64, a.causal = true, a.causal_off = 17;
    ok = sdpa_case("causal 2x33/50 offset 17", a, 2, false, false, false, rng) && ok;
    // Swin: a relative position bias and the shifted windows' masks, hd 32, 49 rows
    a = {}, a.batch = 8, a.heads = 3, a.lq = a.lk = 49, a.hd = 32, a.mask_n = 4;
    ok = sdpa_case("swin 8x49 bias+mask (4)", a, 8, false, true, true, rng) && ok;
    // hd 128 and odd sizes
    a = {}, a.batch = 2, a.heads = 2, a.lq = 70, a.lk = 33, a.hd = 128;
    ok = sdpa_case("rows 2x70/33 d128", a, 2, false, true, false, rng) && ok;
    std::printf("%s\n", ok ? "sdpa: ok" : "sdpa: FAILED");
    return ok ? 0 : 1;
}

void write_raw(const std::string& path, const void* data, size_t bytes) {
    std::ofstream o(path, std::ios::binary);
    o.write(static_cast<const char*>(data), (std::streamsize) bytes);
    if (!o) throw std::runtime_error("cannot write " + path);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "sdpa") return sdpa_selftest();
        if (mode == "sam3" && argc >= 6) {
            si::Sam3 m(argv[2]);
            si::QiRef img;
            std::string err;
            img.rgba = si::load_rgba8(argv[3], img.width, img.height, err);
            if (img.rgba.empty()) throw std::runtime_error(err);
            const auto ids = ints(argv[4]);
            const std::string out = argv[5];
            const float score = argc > 6 ? std::stof(argv[6]) : 0.5f, mask = argc > 7 ? std::stof(argv[7]) : 0.5f;
            m.dump = [&out](const char* name, const std::vector<float>& v) {
                write_raw(out + "/" + name + ".f32", v.data(), v.size() * 4);
            };
            si::Sam3::Times t;
            auto got = m.run(img, ids, score, mask, &t);
            m.dump = nullptr;
            for (size_t i = 0; i < got.size(); ++i) {
                const auto& g = got[i];
                std::printf("segment %zu score=%.4f box=%.1f,%.1f,%.1f,%.1f\n", i, g.score, g.x0, g.y0, g.x1, g.y1);
                write_raw(out + "/mask" + std::to_string(i) + ".u8", g.mask.data(), g.mask.size());
            }
            const char* rep = strata::env_get("PROBE_REPEAT");
            const int reps = rep != nullptr ? (int) std::strtol(rep, nullptr, 10) : 1;
            std::printf("cold vision_s=%.4f text_s=%.4f detect_s=%.4f masks_s=%.4f\n", t.vision, t.text, t.detect,
                        t.masks);
            for (int r = 0; r < reps; ++r) {
                si::Sam3::Times t2;
                got = m.run(img, ids, score, mask, &t2);
                std::printf("warm vision_s=%.4f text_s=%.4f detect_s=%.4f masks_s=%.4f n=%zu held_mib=%llu\n",
                            t2.vision, t2.text, t2.detect, t2.masks, got.size(),
                            (unsigned long long) (m.held() >> 20));
            }
            return 0;
        }
        if (mode == "birefnet" && argc >= 5) {
            si::Birefnet m(argv[2]);
            si::QiRef img;
            std::string err;
            img.rgba = si::load_rgba8(argv[3], img.width, img.height, err);
            if (img.rgba.empty()) throw std::runtime_error(err);
            const std::string out = argv[4];
            const int64_t sw = argc > 5 ? std::stoll(argv[5]) : 1024, sh = argc > 6 ? std::stoll(argv[6]) : sw;
            m.dump = [&out](const char* name, const std::vector<float>& v) {
                write_raw(out + "/" + name + ".f32", v.data(), v.size() * 4);
            };
            si::Birefnet::Times t;
            auto alpha = m.run(img, sw, sh, &t);
            m.dump = nullptr;
            write_raw(out + "/alpha.u8", alpha.data(), alpha.size());
            auto t0 = std::chrono::steady_clock::now();
            const auto fg = m.foreground(img, alpha);
            const double fg_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            write_raw(out + "/fg.rgb", fg.data(), fg.size());
            std::printf("cold backbone_s=%.4f decoder_s=%.4f mask_s=%.4f refine_s=%.4f\n", t.backbone, t.decoder,
                        t.mask, fg_s);
            const char* rep = strata::env_get("PROBE_REPEAT");
            const int reps = rep != nullptr ? (int) std::strtol(rep, nullptr, 10) : 1;
            for (int r = 0; r < reps; ++r) {
                si::Birefnet::Times t2;
                alpha = m.run(img, sw, sh, &t2);
                t0 = std::chrono::steady_clock::now();
                (void) m.foreground(img, alpha);
                std::printf("warm backbone_s=%.4f decoder_s=%.4f mask_s=%.4f refine_s=%.4f held_mib=%llu\n",
                            t2.backbone, t2.decoder, t2.mask,
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                            (unsigned long long) (m.held() >> 20));
            }
            return 0;
        }
        if (mode == "florence" && argc >= 6) {
            si::Florence2 f(argv[2]);
            si::QiRef img;
            std::string err;
            img.rgba = si::load_rgba8(argv[3], img.width, img.height, err);
            if (img.rgba.empty()) throw std::runtime_error(err);
            const auto ids = ints(argv[4]);
            const std::string out = argv[5];
            const int beams = argc > 6 ? std::stoi(argv[6]) : 3, max_new = argc > 7 ? std::stoi(argv[7]) : 1024;
            f.dump = [&out](const char* name, const std::vector<float>& v) {
                write_raw(out + "/" + name + ".f32", v.data(), v.size() * 4);
            };
            si::Florence2::Times t;
            const auto print = [](const std::vector<int32_t>& v) {
                std::printf("ids");
                for (int32_t i : v) std::printf(" %d", i);
                std::printf("\n");
            };
            auto got = f.run(img, ids, max_new, beams, &t);
            f.dump = nullptr;
            print(got);
            // once more (STRATA_PROBE_REPEAT times), warm, for the times (and the same tokens without the dumps' waits)
            si::Florence2::Times t2;
            const char* rep = strata::env_get("PROBE_REPEAT");
            for (int r = 0, n = rep ? std::max(1, (int) std::strtol(rep, nullptr, 10)) : 1; r < n; ++r) {
                got = f.run(img, ids, max_new, beams, &t2);
                print(got);
            }
            std::printf("cold vision_s=%.4f encoder_s=%.4f decoder_s=%.4f steps=%d\n", t.vision, t.encoder,
                        t.decoder, t.steps);
            std::printf("warm vision_s=%.4f encoder_s=%.4f decoder_s=%.4f steps=%d held_mib=%llu\n", t2.vision,
                        t2.encoder, t2.decoder, t2.steps, (unsigned long long) (f.held() >> 20));
            return 0;
        }
        if (mode == "mask" && argc >= 5) {
            std::vector<std::vector<float>> polys;
            for (int i = 5; i < argc; ++i) polys.push_back(floats(argv[i]));
            const int64_t W = std::stoll(argv[2]), H = std::stoll(argv[3]);
            const auto m = si::Florence2::polygon_mask(polys, W, H);
            std::vector<uint8_t> raw((size_t) (W * H));
            for (size_t i = 0; i < raw.size(); ++i) raw[i] = (uint8_t) (255 - m.rgba[i * 4 + 3]);
            write_raw(argv[4], raw.data(), raw.size());
            return 0;
        }
        std::fprintf(stderr, "usage: %s sdpa | sam3 <weights> <image> <ids> <outdir> [score] [mask] | birefnet <weights> <image> <outdir> [width [height]] | florence <model> <image> <ids> <outdir> [beams] [max_new] | mask <W> <H> <out> "
                             "<x,y,...>...\n", argv[0]);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata-image-aux-probe: %s\n", e.what());
        return 1;
    }
}
