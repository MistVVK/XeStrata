// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/sam3_track_probe.cpp - strata-image-sam3-track-probe: SAM 3.1's tracker's parts (sam3.hpp's video
// section) on the GPU against a reference run's calls of the same parts, each fed the run's own inputs: a folder per
// call (<part>_<k>/<name>.npy, FP32, as the records' golden_sam3_track.py and npz_split.py write them).
//
//   strata-image-sam3-track-probe CKPT.safetensors GOLDEN_DIR
//   strata-image-sam3-track-probe video CKPT.safetensors VIDEO IDS OUT_DIR
//       SAM 3.1's video segmentation (sam3_video.hpp) of the words' ids (CLIP's, comma separated) on the video: each
//       frame's objects to OUT_DIR/frame_<i>.bin (their count as int64, then each: id int64, score float32, the mask
//       width x height bytes), the parts' seconds.
//
// Prints each part's cosine and relative L1 against the run's.  Where the engine and the run differ (the sine
// positions' formula, the trunk's input range), the run's own inputs are given, so the parts' arithmetic is what is
// compared.

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/sam3.hpp"
#include "strata/image/sam3_video.hpp"
#include "strata/video/video_io.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/platform/env.hpp"
#include "probe_npy.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace si = strata::image;
namespace gpu = strata::gpu;

namespace {

// [C][H W] -> [H W][C]
std::vector<float> hwc(const std::vector<float>& chw, int64_t C) {
    const int64_t P = (int64_t) chw.size() / C;
    std::vector<float> o(chw.size());
    for (int64_t c = 0; c < C; ++c)
        for (int64_t p = 0; p < P; ++p) o[(size_t) (p * C + c)] = chw[(size_t) (c * P + p)];
    return o;
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t bytes) {
        if (!gpu::alloc_device(&p, std::max<size_t>(bytes, 4))) throw std::runtime_error("GPU memory");
    }
    ~Dev() { gpu::free(p); }
    template<class T> T* as() const { return static_cast<T*>(p); }
};

float* up(Dev& d, const std::vector<float>& v) {
    gpu::copy(d.p, v.data(), v.size() * 4);
    return d.as<float>();
}

si::Sam3* g_sam = nullptr;   // its stream waited for before a read

std::vector<float> down(const float* p, size_t n) {
    if (g_sam != nullptr) g_sam->sync();
    std::vector<float> v(n);
    gpu::copy(v.data(), p, n * 4);
    return v;
}

uint16_t h16(float v) { return strata::kernels::f16_from_f32(v); }

}  // namespace

int video_mode(int argc, char** argv) {
    if (argc != 6 && argc != 7) {
        std::fprintf(stderr, "usage: %s video CKPT.safetensors VIDEO IDS[;IDS...] OUT_DIR [CLICKS]\n"
                     "  CLICKS: frame,object,x,y,label[,x,y,label...][;...] (object -1: a new one)\n", argv[0]);
        return 2;
    }
    try {
        std::vector<si::Sam3VideoWords> words(1);
        for (const char* p = std::strcmp(argv[4], "-") == 0 ? "" : argv[4]; *p != 0;) {
            char* e = nullptr;
            words.back().ids.push_back((int32_t) std::strtol(p, &e, 10));
            if (e == p) throw std::invalid_argument("IDS: numbers split by , and ;");
            if (*e == ';') words.emplace_back();
            p = *e == ',' || *e == ';' ? e + 1 : e;
        }
        if (words.size() == 1 && words[0].ids.empty()) words.clear();   // IDS "-": clicks only
        std::vector<si::Sam3VideoClick> clicks;
        if (argc == 7)
            for (const char* p = argv[6]; *p != 0;) {
                std::vector<float> v;
                for (;;) {
                    char* e = nullptr;
                    v.push_back(std::strtof(p, &e));
                    p = e;
                    if (*p != ',') break;
                    ++p;
                }
                if (v.size() < 5 || (v.size() - 2) % 3 != 0) throw std::invalid_argument("CLICKS");
                si::Sam3VideoClick c;
                c.frame = (int64_t) v[0];
                c.object = (int64_t) v[1];
                for (size_t i = 2; i + 2 < v.size(); i += 3) c.points.push_back({v[i], v[i + 1], v[i + 2]});
                clicks.push_back(c);
                if (*p == ';') ++p;
            }
        const std::string out = argv[5];
        std::filesystem::create_directories(out);
        const strata::video::Info info = strata::video::probe(argv[3]);
        si::Sam3 sam(argv[2]);
        // STRATA_PROBE_REPEAT: the clip run that many times, each one's times printed (the last warm)
        const char* rep = strata::env_get("PROBE_REPEAT");
        int repeat = 1;
        if (rep != nullptr) {
            char* end = nullptr;
            const long n = std::strtol(rep, &end, 10);
            if (end == rep || *end != '\0' || n < 1 || (long) (int) n != n)
                std::fprintf(stderr, "sam3_track_probe: invalid PROBE_REPEAT '%s': using 1\n", rep);
            else
                repeat = (int) n;
        }
        for (int r = 0; r < repeat; ++r) {
            si::Sam3Video vs(sam, info.width, info.height, info.frames, words, clicks);
            strata::video::Reader rd(argv[3], info);
            std::vector<uint8_t> rgb((size_t) rd.frame_bytes());
            const auto write = [&](const std::vector<si::Sam3Video::Frame>& fs) {
                for (const auto& f : fs) {
                    char name[64];
                    std::snprintf(name, sizeof name, "/frame_%04lld.bin", (long long) f.index);
                    std::ofstream o(out + name, std::ios::binary);
                    const auto n = (int64_t) f.objects.size();
                    o.write(reinterpret_cast<const char*>(&n), 8);
                    std::printf("%lld:", (long long) f.index);
                    for (const auto& ob : f.objects) {
                        o.write(reinterpret_cast<const char*>(&ob.id), 8);
                        o.write(reinterpret_cast<const char*>(&ob.score), 4);
                        o.write(reinterpret_cast<const char*>(ob.mask.data()), (std::streamsize) ob.mask.size());
                        std::printf(" %lld", (long long) ob.id);
                    }
                    std::printf("\n");
                }
            };
            int64_t n = 0;
            while (rd.read(rgb.data(), 1) == 1) {
                si::QiRef pic;
                pic.width = info.width;
                pic.height = info.height;
                pic.rgba.resize((size_t) (info.width * info.height * 4));
                for (int64_t p = 0; p < info.width * info.height; ++p) {
                    for (int c = 0; c < 3; ++c) pic.rgba[(size_t) (p * 4 + c)] = rgb[(size_t) (p * 3 + c)];
                    pic.rgba[(size_t) (p * 4 + 3)] = 255;
                }
                write(vs.push(pic));
                ++n;
            }
            write(vs.finish());
            const auto& t = vs.times();
            std::printf("frames=%lld trunk_s=%.3f detect_s=%.3f track_s=%.3f memory_s=%.3f rules_s=%.3f output_s=%.3f\n",
                        (long long) n, t.trunk, t.detect, t.track, t.memory, t.rules, t.output);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sam3-track-probe: %s\n", e.what());
        return 1;
    }
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "video") return video_mode(argc, argv);
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s CKPT.safetensors GOLDEN_DIR\n", argv[0]);
        return 2;
    }
    try {
        const std::string dir = argv[2];
        const auto at = [&](const std::string& part, const std::string& name,
                            std::vector<int64_t>* shape = nullptr) {
            return npy<float>(dir + "/" + part + "/" + name + ".npy", shape);
        };
        const auto has = [&](const std::string& part) { return std::filesystem::exists(dir + "/" + part); };
        si::Sam3 sam(argv[1]);
        sam.track_load();
        g_sam = &sam;
        const int64_t d = 256, G = 72, P = G * G, M = si::Sam3::kMux;
        std::map<std::string, std::vector<std::vector<float>>> dumps;
        const char* dd = std::getenv("SAM3_PROBE_DUMP");   // a folder: each dump as <name>_<k>.f32 there too
        std::map<std::string, int> nth;
        sam.dump = [&](const char* name, const std::vector<float>& v) {
            dumps[name].push_back(v);
            if (dd == nullptr) return;
            const std::string f = std::string(dd) + "/" + name + "_" + std::to_string(nth[name]++) + ".f32";
            std::ofstream(f, std::ios::binary).write(reinterpret_cast<const char*>(v.data()),
                                                     (std::streamsize) (v.size() * 4));
        };
        // the trunk and the propagation neck
        {
            const auto pix = at("trunk_0", "in0");
            sam.trunk_pixels(pix.data());
            const auto want = hwc(at("trunk_0", "out0"), 1024);
            compare("trunk", dumps["vit31"].back().data(), want.data(), want.size());
            sam.track_features();
            for (int i = 0; i < 3; ++i) {
                const auto w = hwc(at("prop" + std::to_string(i) + "_0", "out0"), d);
                const auto g = down(sam.track_level(i), w.size());
                compare(("neck" + std::to_string(i)).c_str(), g.data(), w.data(), w.size());
            }
        }
        // the interactive decoder on the first frame's masks
        {
            std::vector<int64_t> ms;
            const auto masks = at("imds_0", "in0", &ms);
            const int64_t n = ms[0];
            Dev dm(masks.size() * 4), dt((size_t) (n * d) * 4);
            dumps.clear();
            std::vector<float> score;
            sam.interactive(up(dm, masks), n, dt.as<float>(), score);
            const auto ids = at("imds_0", "out0");
            std::vector<float> g;
            for (const auto& v : dumps["imds"]) g.insert(g.end(), v.begin(), v.end());
            compare("imds", g.data(), ids.data(), ids.size());
            const auto dense = at("ipenc_0", "out1");
            g.clear();
            for (const auto& v : dumps["prompt_dense"]) g.insert(g.end(), v.begin(), v.end());
            std::vector<float> want;
            for (int64_t i = 0; i < n; ++i) {
                const std::vector<float> one(dense.begin() + i * d * P, dense.begin() + (i + 1) * d * P);
                const auto h = hwc(one, d);
                want.insert(want.end(), h.begin(), h.end());
            }
            compare("prompt", g.data(), want.data(), want.size());
            const auto tok = at("idec_0", "out2"), sc = at("idec_0", "out3");
            const auto gt = down(dt.as<float>(), (size_t) (n * d));
            compare("itokens", gt.data(), tok.data(), tok.size());
            compare("iscores", score.data(), sc.data(), sc.size());
            std::printf("           scores");
            for (int64_t i = 0; i < n; ++i) std::printf(" %.3f/%.3f", score[(size_t) i], sc[(size_t) i]);
            std::printf("\n");
        }
        // the pointers' projections (the interactive's, the propagation's)
        for (const char* part : {"iptrproj_0", "ptrproj_0"}) {
            if (!has(part)) continue;
            std::vector<int64_t> s;
            const auto in = at(part, "in0", &s);
            const auto want = at(part, "out0");
            Dev di(in.size() * 4), dp(in.size() * 4);
            // all seen (the projection alone: the run's no-object blend comes after the hooked call)
            sam.pointers(up(di, in), s[0], std::string(part) == "iptrproj_0", std::vector<float>((size_t) s[0], 1.f),
                         dp.as<float>());
            const auto g = down(dp.as<float>(), in.size());
            compare(part, g.data(), want.data(), want.size());
        }
        // the memory attention, each call
        for (int k = 0; has("memattn_" + std::to_string(k)); ++k) {
            const std::string c = "memattn_" + std::to_string(k);
            std::vector<int64_t> xs, ms, is;
            const auto x = at(c, "x", &xs);
            const auto mem = at(c, "memory", &ms);
            const auto mpos = at(c, "memory_pos");
            const auto spos = at(c, "src_pos");
            const auto mimg = at(c, "memory_image", &is);
            const auto mipos = at(c, "memory_image_pos");
            const int64_t N = ms[1], nsp = is[1];
            std::vector<uint16_t> m16((size_t) (N * 2 * d), 0);
            for (int64_t r = 0; r < N; ++r)
                for (int64_t j = 0; j < d; ++j) {
                    if (r < nsp) m16[(size_t) (r * 2 * d + j)] = h16(mimg[(size_t) (r * d + j)]);
                    m16[(size_t) (r * 2 * d + d + j)] = h16(mem[(size_t) (r * d + j)]);
                }
            const std::vector<float> ptr_pos(mpos.begin() + nsp * d, mpos.end());
            Dev dx(x.size() * 4), dsp(spos.size() * 4), dm(m16.size() * 2), dsine(mipos.size() * 4),
                dptr(std::max<size_t>(ptr_pos.size(), 1) * 4), dout(x.size() * 4);
            gpu::copy(dm.p, m16.data(), m16.size() * 2);
            sam.mem_attention(up(dx, x), up(dsp, spos), dm.as<uint16_t>(), 1, N, nsp, up(dsine, mipos), nullptr,
                              ptr_pos.empty() ? nullptr : up(dptr, ptr_pos), 1, dout.as<float>());
            const auto g = down(dout.as<float>(), x.size());
            const auto want = at(c, "out0");
            compare(c.c_str(), g.data(), want.data(), want.size());
        }
        // the multiplex decoder, each call (its slots valid as its objects)
        for (int k = 0; has("mdec_" + std::to_string(k)); ++k) {
            const std::string c = "mdec_" + std::to_string(k);
            const auto img = hwc(at(c, "image_embeddings"), d);
            const auto hr0 = hwc(at(c, "high_res_features_0"), d), hr1 = hwc(at(c, "high_res_features_1"), d);
            // the slots valid as the objects (the first frame's masks), the rest not
            std::vector<uint8_t> valid((size_t) M);
            {
                std::vector<int64_t> s;
                (void) at("imds_0", "in0", &s);
                for (int64_t i = 0; i < M; ++i) valid[(size_t) i] = i < s[0];
            }
            Dev di(img.size() * 4), d0(hr0.size() * 4), d1(hr1.size() * 4);
            const auto r = sam.mux_decode(up(di, img), up(d0, hr0), up(d1, hr1), valid, 1);
            const auto gm = down(r.masks, (size_t) (M * r.T * r.P0));
            const auto gi = down(r.iou, (size_t) (M * r.T));
            const auto gs = down(r.score, (size_t) M);
            const auto gt = down(r.tokens, (size_t) (M * r.T * d));
            const auto wm = at(c, "out0"), wi = at(c, "out1"), wt = at(c, "out2"), ws = at(c, "out3");
            compare((c + " masks").c_str(), gm.data(), wm.data(), wm.size());
            compare((c + " iou").c_str(), gi.data(), wi.data(), wi.size());
            compare((c + " score").c_str(), gs.data(), ws.data(), ws.size());
            std::vector<float> t0((size_t) (M * d));
            for (int64_t s = 0; s < M; ++s)
                for (int64_t j = 0; j < d; ++j) t0[(size_t) (s * d + j)] = gt[(size_t) ((s * r.T) * d + j)];
            compare((c + " tok0").c_str(), t0.data(), wt.data(), wt.size());
        }
        // the memory encoder, each call fed its own 1152 x 1152 input (the cond frame's 1008 one is resized inside)
        for (int k = 1; has("memenc_" + std::to_string(k)); ++k) {
            const std::string c = "memenc_" + std::to_string(k);
            const auto pix = hwc(at(c, "in0"), d);
            std::vector<int64_t> s;
            const auto mux = at(c, "in1", &s);
            const int64_t C = s[1], S = s[2];
            std::vector<uint16_t> m16((size_t) (S * S * C));
            for (int64_t ch = 0; ch < C; ++ch)
                for (int64_t p = 0; p < S * S; ++p) m16[(size_t) (p * C + ch)] = h16(mux[(size_t) (ch * S * S + p)]);
            Dev dp(pix.size() * 4), dm(m16.size() * 2), dout(pix.size() * 4);
            gpu::copy(dm.p, m16.data(), m16.size() * 2);
            sam.mem_encode_muxed(up(dp, pix), dm.as<uint16_t>(), std::vector<float>((size_t) M, 1.f), dout.as<float>());
            const auto g = down(dout.as<float>(), pix.size());
            const auto want = hwc(at(c, "out_vision_features"), d);
            compare(c.c_str(), g.data(), want.data(), want.size());
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sam3-track-probe: %s\n", e.what());
        return 1;
    }
}
