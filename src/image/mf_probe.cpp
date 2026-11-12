// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/mf_probe.cpp - strata-image-mf-probe: Mage-Flow's parts on the GPU against the reference implementation's
// outputs of one run (the .npy files of records' mage-flow-golden/golden.py).
//
//   strata-image-mf-probe text TEXT_ENCODER.gguf GOLDEN_DIR [MMPROJ.gguf]   (an edit's run: its images vl0.png ..
//       through the vision encoder, the system turn 64 ids)
//   strata-image-mf-probe vae VAE.safetensors GOLDEN_DIR [OUT.f32]   (golden_vae.py's vae_cond, vae_s, vae_img;
//       the Flux2 VAE: golden_f2vae.py's f2_img of the run's latent, whole and in tiles, the tiled image as FP32
//       [3][H][W] to OUT.f32)
//   strata-image-mf-probe vae-enc VAE.safetensors GOLDEN_DIR   (golden_edit.py's ref0_pix: its posterior's mean,
//       log-variance and, with the run's own noise, the sample; the Flux2 VAE: golden_f2enc.py's ref0_z, the latent
//       of ref0_pix as FLUX.2's pipelines take it)
//   strata-image-mf-probe dit DIT.safetensors GOLDEN_DIR [q8|bf16 [VAE.safetensors OUT.f32]]
//       (block 0, the first velocity, the final latent; with the VAE the image as FP32 [3][H][W] to OUT.f32)
//
// Prints how far each part's output is from the run's (cosine and relative L1) and its time.

#include "strata/core/runtime.hpp"
#include "strata/core/gpu.hpp"
#include "strata/image/mf_dit.hpp"
#include "strata/image/mf_vae.hpp"
#include "strata/image/qi_image_io.hpp"
#include "strata/core/vision.hpp"
#include "strata/image/text_encoder.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// a .npy file's data as T (its dtype '<f4' or '<i8', C order), and its shape
template<class T> std::vector<T> npy(const std::string& path, std::vector<int64_t>* shape = nullptr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error(path + ": not a .npy file");
    uint32_t hlen = 0;
    if (magic[6] == 1) {
        uint16_t h = 0;
        f.read(reinterpret_cast<char*>(&h), 2);
        hlen = h;
    } else {
        f.read(reinterpret_cast<char*>(&hlen), 4);
    }
    std::string header(hlen, '\0');
    f.read(header.data(), hlen);
    const char* want = sizeof(T) == 8 ? "'<i8'" : "'<f4'";
    if (header.find(want) == std::string::npos || header.find("'fortran_order': False") == std::string::npos)
        throw std::runtime_error(path + ": expected " + want + " in C order: " + header);
    const size_t l = header.find('('), r = header.find(')');
    std::vector<int64_t> dims;
    size_t n = 1;
    for (size_t i = l + 1; i < r;) {
        size_t j = header.find_first_of(",)", i);
        const std::string tok = header.substr(i, j - i);
        if (tok.find_first_not_of(' ') != std::string::npos) {
            dims.push_back(std::stoll(tok));
            n *= (size_t) dims.back();
        }
        i = j + 1;
    }
    std::vector<T> v(n);
    f.read(reinterpret_cast<char*>(v.data()), (std::streamsize) (n * sizeof(T)));
    if (!f) throw std::runtime_error(path + ": short");
    if (shape) *shape = dims;
    return v;
}

void compare(const char* what, const float* got, const float* want, size_t n) {
    double dot = 0, ng = 0, nw = 0, err = 0, mag = 0;
    for (size_t i = 0; i < n; ++i) {
        dot += (double) got[i] * want[i];
        ng += (double) got[i] * got[i];
        nw += (double) want[i] * want[i];
        err += std::fabs((double) got[i] - want[i]);
        mag += std::fabs((double) want[i]);
    }
    std::printf("%-10s cos %.6f  rel %.3g  (%zu values)\n", what, dot / std::sqrt(ng * nw), err / mag, n);
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// the text encoder: each prompt of the run (ids.npy, ids_lens.npy) encoded alone, its rows after the 34 of the
// system turn against txt.npy
int text(const std::string& gguf, const std::string& dir, const std::string& mmproj = "") {
    const int64_t kDrop = mmproj.empty() ? 34 : 64;   // the system turn: generation's, an edit's
    void* s = strata::core::Runtime::get().create_stream();
    strata::image::TextEncoder te;
    std::string err;
    if (!te.open(gguf, 16384, err)) throw std::runtime_error(err);
    const int32_t kImagePad = te.image_pad();   // the <|image_pad|> token, from the vocabulary
    te.set_final_norm(true);
    te.set_flat_positions(true);
    // an edit's images (as the processor took them: their sides the grid's 16-pixel patches) through the vision encoder
    std::vector<strata::core::VisionOut> vouts;
    if (!mmproj.empty()) {
        strata::core::Vision vis;
        if (!vis.open(mmproj, s, err)) throw std::runtime_error(err);
        for (int j = 0;; ++j) {
            const std::string f = dir + "/vl" + std::to_string(j) + ".png";
            if (!std::ifstream(f)) break;
            int64_t w = 0, h = 0;
            const auto px = strata::image::load_rgba8(f, w, h, err);
            if (px.empty()) throw std::runtime_error(err);
            std::vector<float> rgb((size_t) (3 * w * h));
            for (int64_t i = 0; i < w * h; ++i)
                for (int c = 0; c < 3; ++c) rgb[(size_t) (c * w * h + i)] = (float) px[(size_t) (i * 4 + c)] / 127.5f - 1.f;
            const auto t0 = std::chrono::steady_clock::now();
            vouts.push_back(vis.encode(rgb.data(), w, h));
            std::printf("vision %lld x %lld: %lld tokens, %.1f ms\n", (long long) w, (long long) h,
                        (long long) vouts.back().rows, 1e3 * seconds_since(t0));
            const std::string vj = dir + "/vit" + std::to_string(j);
            if (std::ifstream(vj + "_main.npy")) {   // vit_dump.py's outputs (bf16, and FP32)
                const auto& v = vouts.back();
                const auto m = npy<float>(vj + "_main.npy");
                compare("vit main", v.main.data(), m.data(), std::min(m.size(), v.main.size()));
                if (std::ifstream(vj + "_main32.npy")) {
                    const auto m32 = npy<float>(vj + "_main32.npy");
                    compare("vit fp32", v.main.data(), m32.data(), std::min(m32.size(), v.main.size()));
                }
                for (size_t k = 0; k < v.deepstack.size() && std::ifstream(vj + "_ds" + std::to_string(k) + ".npy"); ++k) {
                    const auto d = npy<float>(vj + "_ds" + std::to_string(k) + ".npy");
                    compare(("vit ds" + std::to_string(k)).c_str(), v.deepstack[k].data(), d.data(),
                            std::min(d.size(), v.deepstack[k].size()));
                }
            }
        }
    }
    const auto ids = npy<int64_t>(dir + "/ids.npy");
    const auto lens = npy<int64_t>(dir + "/ids_lens.npy");
    std::vector<int64_t> shape;
    const auto want = npy<float>(dir + "/txt.npy", &shape);
    const int64_t C = te.hidden();
    if (shape.size() != 2 || shape[1] != C) throw std::runtime_error("txt.npy is not [rows][hidden]");
    std::vector<float> got;
    size_t off = 0;
    for (int64_t len : lens) {
        std::vector<int32_t> p(ids.begin() + (long) off, ids.begin() + (long) (off + len));
        off += (size_t) len;
        std::vector<strata::image::TextEncoder::Picture> ims;   // the prompt's runs of image pads, image by image
        for (size_t i = 0, k = 0; i < p.size(); ++i) {
            if (p[i] != kImagePad || (i > 0 && p[i - 1] == kImagePad)) continue;
            if (k >= vouts.size()) throw std::runtime_error("image pads beyond the images (pass the mmproj)");
            const auto& v = vouts[k++];
            strata::image::TextEncoder::Picture im;
            im.index = (int64_t) i;
            im.nx = v.grid_w;
            im.ny = v.grid_h;
            im.rows = v.main.data();
            for (const auto& d : v.deepstack) im.deepstack.push_back(d.data());
            ims.push_back(im);
        }
        strata::gpu::stream_sync(s);   // the vision encoder's work first (the text encoder runs on its own stream)
        te.encode(p, err, ims);   // warm
        const auto t0 = std::chrono::steady_clock::now();
        const auto h = te.encode(p, err, ims);
        if (h.empty()) throw std::runtime_error(err);
        std::printf("encode %lld tokens: %.1f ms\n", (long long) len, 1e3 * seconds_since(t0));
        got.insert(got.end(), h.begin() + kDrop * C, h.end());
    }
    if ((int64_t) got.size() != shape[0] * C) throw std::runtime_error("the rows differ from txt.npy's");
    compare("text", got.data(), want.data(), got.size());
    if (std::ifstream(dir + "/txt32.npy")) {   // text_fp32.py's states in FP32
        const auto t32 = npy<float>(dir + "/txt32.npy");
        compare("text fp32", got.data(), t32.data(), std::min(t32.size(), got.size()));
    }
    return 0;
}

// [C][HW] (the reference's channels-first) of a [HW][C] tensor
std::vector<float> channels_first(const std::vector<float>& x, int64_t HW, int64_t C) {
    std::vector<float> y(x.size());
    for (int64_t p = 0; p < HW; ++p)
        for (int64_t c = 0; c < C; ++c) y[(size_t) (c * HW + p)] = x[(size_t) (p * C + c)];
    return y;
}

std::vector<float> fetch(const float* d, size_t n) {
    std::vector<float> h(n);
    strata::gpu::copy(h.data(), d, n * 4);
    return h;
}

float* put(const std::vector<float>& h) {
    float* d = nullptr;
    if (!strata::gpu::alloc_device(&d, h.size() * 4)) throw std::runtime_error("GPU memory");
    strata::gpu::copy(d, h.data(), h.size() * 4);
    return d;
}

// the VAE's decode of the run's latent, its condition, blocks' state and image against golden_vae.py's
// the image's PSNR (8-bit) against the reference's, both clamped to -1 .. 1
void psnr(const std::vector<float>& got, const std::vector<float>& img) {
    double se = 0;
    for (size_t i = 0; i < img.size(); ++i) {
        const double a = std::min(1.0, std::max(-1.0, (double) got[i]));
        const double b = std::min(1.0, std::max(-1.0, (double) img[i]));
        se += (a - b) * (a - b) * 127.5 * 127.5;
    }
    std::printf("image PSNR %.2f dB (8-bit)\n", 10 * std::log10(255.0 * 255.0 / (se / (double) img.size())));
}

int vae(const std::string& path, const std::string& dir, const std::string& out_path = "") {
    void* s = strata::core::Runtime::get().create_stream();
    strata::image::MfVae v;
    std::string err;
    if (!v.open(path, s, err)) throw std::runtime_error(err);
    std::vector<int64_t> shape;
    const auto lat = npy<float>(dir + "/latent.npy", &shape);   // [1][h w][128]
    if (v.flux2()) {   // the image alone: [1][3][H][W]
        std::vector<int64_t> is;
        const auto img = npy<float>(dir + "/f2_img.npy", &is);
        const int64_t h = is[2] / 16, w = is[3] / 16;
        if (shape.size() != 3 || shape[1] != h * w) throw std::runtime_error("latent.npy is not [1][h w][128]");
        float* z = put(lat);
        float* rgb = nullptr;
        if (!strata::gpu::alloc_device(&rgb, img.size() * 4)) throw std::runtime_error("GPU memory");
        v.decode_raw(z, h, w, rgb);   // warm
        strata::gpu::stream_sync(s);
        const auto t0 = std::chrono::steady_clock::now();
        v.decode_raw(z, h, w, rgb);
        strata::gpu::stream_sync(s);
        std::printf("decode %lld x %lld: %.1f ms (Flux2 VAE, scratch %.0f MiB)\n", 16 * (long long) h,
                    16 * (long long) w, 1e3 * seconds_since(t0), (double) v.scratch_bytes(h, w) / 1048576.0);
        const auto got = fetch(rgb, img.size());
        compare("image", got.data(), img.data(), img.size());
        psnr(got, img);
        if (!out_path.empty()) {   // the whole's decode beside the tiled one's
            std::ofstream o(out_path + ".whole", std::ios::binary);
            o.write(reinterpret_cast<const char*>(got.data()), (std::streamsize) (got.size() * 4));
        }
        // in tiles: decode within a quarter of the whole's scratch (rgba 0 .. 1, back to -1 .. 1 for the PSNR)
        const uint64_t limit = v.scratch_bytes(h, w) / 4;
        float* rgba = nullptr;
        if (!strata::gpu::alloc_device(&rgba, img.size() / 3 * 4 * 4)) throw std::runtime_error("GPU memory");
        v.drop_scratch();
        const auto t1 = std::chrono::steady_clock::now();
        v.decode(z, h, w, rgba, limit);
        strata::gpu::stream_sync(s);
        std::printf("tiled within %.0f MiB: %.1f ms, scratch held %.0f MiB\n", (double) limit / 1048576.0,
                    1e3 * seconds_since(t1), (double) v.scratch_held() / 1048576.0);
        auto tiled = fetch(rgba, img.size());
        for (float& x : tiled) x = 2.f * x - 1.f;
        psnr(tiled, img);
        if (!out_path.empty()) {
            std::ofstream o(out_path, std::ios::binary);
            o.write(reinterpret_cast<const char*>(tiled.data()), (std::streamsize) (tiled.size() * 4));
        }
        return 0;
    }
    std::vector<int64_t> cs;
    const auto cond = npy<float>(dir + "/vae_cond.npy", &cs);   // [1][384][h][w]
    const int64_t h = cs[2], w = cs[3], C = cs[1], HW = h * w;
    if (shape.size() != 3 || shape[1] != HW) throw std::runtime_error("latent.npy is not [1][h w][128]");
    float* z = put(lat);
    float* rgb = nullptr;
    if (!strata::gpu::alloc_device(&rgb, (size_t) (3 * HW * 256) * 4)) throw std::runtime_error("GPU memory");
    v.decode_raw(z, h, w, rgb);   // warm
    strata::gpu::stream_sync(s);
    const auto t0 = std::chrono::steady_clock::now();
    v.decode_raw(z, h, w, rgb);
    strata::gpu::stream_sync(s);
    std::printf("decode %lld x %lld: %.1f ms\n", 16 * (long long) h, 16 * (long long) w, 1e3 * seconds_since(t0));
    compare("cond", channels_first(fetch(v.last_cond(), (size_t) (HW * C)), HW, C).data(), cond.data(), cond.size());
    const auto st = npy<float>(dir + "/vae_s.npy");
    compare("blocks", channels_first(fetch(v.last_state(), (size_t) (HW * C)), HW, C).data(), st.data(), st.size());
    const auto img = npy<float>(dir + "/vae_img.npy");
    const auto got = fetch(rgb, img.size());
    compare("image", got.data(), img.data(), img.size());
    psnr(got, img);
    return 0;
}

// the DiT from the run's text states, noise and sigmas: block 0's streams and the first velocity, then the whole
// sampling's latent
int dit(const std::string& path, const std::string& dir, bool q8, const std::string& vae_path = "",
        const std::string& out_path = "") {
    void* s = strata::core::Runtime::get().create_stream();
    strata::image::MfDit d;
    std::string err;
    auto t0 = std::chrono::steady_clock::now();
    if (!d.open(path, s, err, false, q8)) throw std::runtime_error(err);
    std::printf("open (%s): %.1f s\n", q8 ? "Q8_0" : "FP16", seconds_since(t0));
    std::vector<int64_t> ns, ts;
    const auto noise = npy<float>(dir + "/noise.npy", &ns);   // [1][128][h][w]
    const int64_t Cl = ns[1], h = ns[2], w = ns[3], HW = h * w, C = d.hidden();
    const auto txt = npy<float>(dir + "/txt.npy", &ts);
    const auto lens = npy<int64_t>(dir + "/txt_lens.npy");
    const auto sig = npy<float>(dir + "/sigmas.npy");
    std::ifstream af(dir + "/args.json");
    const std::string args((std::istreambuf_iterator<char>(af)), std::istreambuf_iterator<char>());
    const size_t ci = args.find("\"cfg\":");
    const float cfg = ci == std::string::npos ? 1.f : std::stof(args.substr(ci + 6));
    const int64_t D = ts[1];
    d.set_context(0, txt.data(), lens[0]);
    if (lens.size() > 1) d.set_context(1, txt.data() + lens[0] * D, lens[1]);
    const int n = cfg > 1.f && lens.size() > 1 ? 2 : 1;
    // an edit's run (golden_edit.py): its references' latent tokens as it sampled them
    std::vector<float> rz;
    std::vector<std::pair<int64_t, int64_t>> rsz;
    for (int j = 0; std::ifstream(dir + "/ref" + std::to_string(j) + "_z.npy"); ++j) {
        std::vector<int64_t> zs;
        const auto z = npy<float>(dir + "/ref" + std::to_string(j) + "_z.npy", &zs);   // [1][128][h][w]
        const int64_t rh = zs[2], rw = zs[3];
        for (int64_t p = 0; p < rh * rw; ++p)
            for (int64_t c = 0; c < zs[1]; ++c) rz.push_back(z[(size_t) (c * rh * rw + p)]);
        rsz.emplace_back(rh, rw);
    }
    if (!rsz.empty()) {
        float* dz = put(rz);
        d.set_refs(dz, rsz);
        std::printf("edit: %zu reference%s, %lld rows\n", rsz.size(), rsz.size() > 1 ? "s" : "",
                    (long long) d.ref_rows());
    }
    std::vector<float> xt((size_t) (HW * Cl));   // the noise as tokens [h w][128]
    for (int64_t c = 0; c < Cl; ++c)
        for (int64_t p = 0; p < HW; ++p) xt[(size_t) (p * Cl + c)] = noise[(size_t) (c * HW + p)];
    float* x = put(xt);
    float* out = nullptr;
    if (!strata::gpu::alloc_device(&out, (size_t) (n * HW * Cl) * 4)) throw std::runtime_error("GPU memory");
    d.keep_block0(true);
    d.forward(x, sig[0], h, w, n, out);
    strata::gpu::stream_sync(s);
    t0 = std::chrono::steady_clock::now();
    d.forward(x, sig[0], h, w, n, out);
    strata::gpu::stream_sync(s);
    std::printf("forward (%d context%s): %.1f ms\n", n, n > 1 ? "s" : "", 1e3 * seconds_since(t0));
    if (std::ifstream(dir + "/blk0_img.npy")) {
        const auto bi = npy<float>(dir + "/blk0_img.npy"), bt = npy<float>(dir + "/blk0_txt.npy");
        compare("blk0 img", fetch(d.block0_img(), (size_t) (HW * C)).data(), bi.data(), (size_t) (HW * C));
        compare("blk0 txt", fetch(d.block0_txt(), (size_t) (lens[0] * C)).data(), bt.data(), (size_t) (lens[0] * C));
    }
    auto o = fetch(out, (size_t) (n * HW * Cl));
    if (n == 2)
        for (size_t i = 0; i < (size_t) (HW * Cl); ++i) o[i] = o[HW * Cl + i] + cfg * (o[i] - o[HW * Cl + i]);
    const auto v0 = npy<float>(dir + "/v0.npy");
    compare("v0", o.data(), v0.data(), v0.size());
    // each later step's velocity from the reference's own latent there (its noise and velocities before), so a step's
    // error is not the sum of those before it
    std::vector<float> xr = xt;
    for (size_t k = 1; k + 1 < sig.size(); ++k) {
        const auto vp = npy<float>(dir + "/v" + std::to_string(k - 1) + ".npy");
        for (size_t i = 0; i < xr.size(); ++i) xr[i] += (sig[k] - sig[k - 1]) * vp[i];
        strata::gpu::copy(x, xr.data(), xr.size() * 4);
        d.forward(x, sig[k], h, w, n, out);
        strata::gpu::stream_sync(s);
        o = fetch(out, (size_t) (n * HW * Cl));
        if (n == 2)
            for (size_t i = 0; i < (size_t) (HW * Cl); ++i) o[i] = o[HW * Cl + i] + cfg * (o[i] - o[HW * Cl + i]);
        const auto vk = npy<float>(dir + "/v" + std::to_string(k) + ".npy");
        compare(("v" + std::to_string(k)).c_str(), o.data(), vk.data(), vk.size());
    }
    strata::gpu::copy(x, xt.data(), xt.size() * 4);
    std::vector<float> sigmas(sig.begin(), sig.end());
    t0 = std::chrono::steady_clock::now();
    d.sample(x, h, w, sigmas, cfg);
    strata::gpu::stream_sync(s);
    std::printf("sample %zu steps: %.2f s\n", sigmas.size() - 1, seconds_since(t0));
    const auto lat = npy<float>(dir + "/latent.npy");
    compare("latent", fetch(x, (size_t) (HW * Cl)).data(), lat.data(), lat.size());
    if (!vae_path.empty()) {   // the image: rgb [3][16 h][16 w] FP32 (about -1 .. 1) to out_path
        strata::image::MfVae v;
        if (!v.open(vae_path, s, err)) throw std::runtime_error(err);
        float* rgb = nullptr;
        if (!strata::gpu::alloc_device(&rgb, (size_t) (3 * HW * 256) * 4)) throw std::runtime_error("GPU memory");
        v.decode_raw(x, h, w, rgb);
        strata::gpu::stream_sync(s);
        const auto img = fetch(rgb, (size_t) (3 * HW * 256));
        std::ofstream(out_path, std::ios::binary).write(reinterpret_cast<const char*>(img.data()),
                                                         (std::streamsize) (img.size() * 4));
        std::printf("image: %s\n", out_path.c_str());
    }
    return 0;
}

}  // namespace

// the VAE's encoder on golden_edit.py's first reference: the posterior's moments, and the sample with the run's noise
int vae_enc(const std::string& path, const std::string& dir) {
    void* s = strata::core::Runtime::get().create_stream();
    strata::image::MfVae v;
    std::string err;
    if (!v.open(path, s, err) || !v.open_encoder(err)) throw std::runtime_error(err);
    std::vector<int64_t> ps, ms;
    const auto pix = npy<float>(dir + "/ref0_pix.npy", &ps);    // [3][H][W]
    const auto mean = npy<float>(dir + "/ref0_mean.npy", &ms);  // [1][128][h][w]
    const auto zg = npy<float>(dir + "/ref0_z.npy");
    const int64_t Z = ms[1], h = ms[2], w = ms[3], HW = h * w;
    float* rgb = put(pix);
    float* z = nullptr;
    if (!strata::gpu::alloc_device(&z, (size_t) (HW * Z) * 4)) throw std::runtime_error("GPU memory");
    if (v.flux2()) {   // golden_f2enc.py's latent: the posterior's mode, patchified and normalized as FLUX.2 takes it
        v.encode(rgb, h, w, nullptr, z);   // warm
        strata::gpu::stream_sync(s);
        const auto t0 = std::chrono::steady_clock::now();
        v.encode(rgb, h, w, nullptr, z);
        strata::gpu::stream_sync(s);
        std::printf("encode %lld x %lld: %.1f ms (Flux2 VAE, scratch %.0f MiB)\n", (long long) ps[2],
                    (long long) ps[1], 1e3 * seconds_since(t0), (double) v.encode_scratch_bytes(h, w) / 1048576.0);
        const auto zt = fetch(z, (size_t) (HW * Z));
        std::vector<float> zc((size_t) (HW * Z));
        for (int64_t p = 0; p < HW; ++p)
            for (int64_t c = 0; c < Z; ++c) zc[(size_t) (c * HW + p)] = zt[(size_t) (p * Z + c)];
        compare("latent", zc.data(), zg.data(), zg.size());
        return 0;
    }
    const auto lv = npy<float>(dir + "/ref0_logvar.npy");
    float* mom = nullptr;
    if (!strata::gpu::alloc_device(&mom, (size_t) (HW * 2 * Z) * 4)) throw std::runtime_error("GPU memory");
    v.encode_moments(rgb, h, w, mom);   // warm
    const auto t0 = std::chrono::steady_clock::now();
    v.encode_moments(rgb, h, w, mom);
    std::printf("encode %lld x %lld: %.1f ms\n", (long long) ps[2], (long long) ps[1], 1e3 * seconds_since(t0));
    const auto m = fetch(mom, (size_t) (HW * 2 * Z));
    std::vector<float> gm((size_t) (HW * Z)), gl((size_t) (HW * Z));
    for (int64_t p = 0; p < HW; ++p)
        for (int64_t c = 0; c < Z; ++c) {
            gm[(size_t) (c * HW + p)] = m[(size_t) (p * 2 * Z + c)];
            gl[(size_t) (c * HW + p)] = std::min(10.f, std::max(-20.f, m[(size_t) (p * 2 * Z + Z + c)]));
        }
    compare("mean", gm.data(), mean.data(), mean.size());
    compare("logvar", gl.data(), lv.data(), lv.size());
    if (std::ifstream(dir + "/ref0_mean32.npy")) {   // enc_fp32.py's moments in FP32
        const auto m32 = npy<float>(dir + "/ref0_mean32.npy");
        compare("mean fp32", gm.data(), m32.data(), m32.size());
    }
    // the run's noise, eps = (z - mean) / std, through the engine's posterior: the sample as the run took it
    std::vector<float> eps((size_t) (HW * Z));
    for (int64_t p = 0; p < HW; ++p)
        for (int64_t c = 0; c < Z; ++c) {
            const size_t g = (size_t) (c * HW + p);
            eps[(size_t) (p * Z + c)] = (zg[g] - mean[g]) / std::exp(0.5f * lv[g]);
        }
    float* e = put(eps);
    v.encode(rgb, h, w, e, z);
    const auto zt = fetch(z, (size_t) (HW * Z));
    std::vector<float> zc((size_t) (HW * Z));
    for (int64_t p = 0; p < HW; ++p)
        for (int64_t c = 0; c < Z; ++c) zc[(size_t) (c * HW + p)] = zt[(size_t) (p * Z + c)];
    compare("sample", zc.data(), zg.data(), zg.size());
    return 0;
}

int main(int argc, char** argv) {
    try {
        const std::string what = argc > 1 ? argv[1] : "";
        if (what == "text" && (argc == 4 || argc == 5)) return text(argv[2], argv[3], argc == 5 ? argv[4] : "");
        if (what == "vae" && (argc == 4 || argc == 5)) return vae(argv[2], argv[3], argc == 5 ? argv[4] : "");
        if (what == "vae-enc" && argc == 4) return vae_enc(argv[2], argv[3]);
        if (what == "dit" && (argc == 4 || argc == 5 || argc == 7))
            return dit(argv[2], argv[3], argc == 4 || std::string(argv[4]) != "bf16", argc == 7 ? argv[5] : "",
                       argc == 7 ? argv[6] : "");
        std::fprintf(stderr, "usage: %s text TEXT_ENCODER.gguf GOLDEN_DIR | vae VAE.safetensors GOLDEN_DIR [OUT.f32] | "
                     "vae-enc VAE.safetensors GOLDEN_DIR | "
                     "dit DIT.safetensors GOLDEN_DIR [q8|bf16 [VAE.safetensors OUT.f32]]\n", argv[0]);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata-image-mf-probe: %s\n", e.what());
        return 1;
    }
}
