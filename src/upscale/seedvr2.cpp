// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/upscale/seedvr2.cpp - see include/strata/upscale/seedvr2.hpp.

#include "strata/upscale/seedvr2.hpp"

#include "strata/core/device.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/mf_vae.hpp"
#include "strata/image/moss_pipeline.hpp"
#include "strata/upscale/seedvr2_dit.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <memory>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::upscale {

namespace seedvr2_detail {

sycl::queue& q_of(void* s) { return core::Runtime::get().stream(s); }

// a frame's RGB in -1 .. 1, [3][H][W] FP32, from RGBA8 [H][W][4]
void rgb_of(const uint8_t* rgba, int64_t HW, float* rgb, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (3 * HW)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t c = i / HW, p = i - c * HW;
        rgb[i] = (float) rgba[p * 4 + c] * (2.f / 255.f) - 1.f;
    });
}

// x = a x
void scale(float* x, int64_t n, float a, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) { x[id[0]] *= a; });
}

// z = (noise - v) a: the step's result, as the decoder takes it
void restore(const float* noise, const float* v, int64_t n, float a, float* z, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = id[0];
        z[i] = (noise[i] - v[i]) * a;
    });
}

// the decoder's planes [4][H][W] (0 .. 1) to RGBA8, the alpha the input's
void rgba8_of(const float* planes, const uint8_t* in, int64_t HW, uint8_t* out, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) HW), [=](sycl::id<1> id) {
        const auto p = static_cast<int64_t>(id[0]);
        for (int c = 0; c < 3; ++c)
            out[p * 4 + c] = (uint8_t) sycl::round(sycl::clamp(planes[c * HW + p], 0.f, 1.f) * 255.f);
        out[p * 4 + 3] = in[p * 4 + 3];
    });
}

}  // namespace seedvr2_detail

using namespace seedvr2_detail;

struct Seedvr2::Impl {
    void* s = nullptr;
    std::string dit_path, vae_path;
    std::unique_ptr<image::MfVae> vae;   // null until a run needs them, and after release(true)
    std::unique_ptr<Seedvr2Dit> dit;
    uint64_t vae_bytes = 0, dit_bytes = 0, dit_least = 0;   // the DiT's: all its blocks held, or two slots of them
    uint64_t room = UINT64_MAX;                           // the next run's VRAM (set_room)
    double enc_s = 0.0, dit_s = 0.0, dec_s = 0.0;         // the last run's parts
    std::vector<void*> scratch;
    uint64_t scratch_bytes = 0;
    void load() {
        if (vae && dit) return;
        std::string err;
        auto v = std::make_unique<image::MfVae>();
        if (!v->open(vae_path, s, err) || !v->open_encoder(err))
            throw std::runtime_error("SeedVR2's VAE (" + vae_path + "): " + err);
        auto d = std::make_unique<Seedvr2Dit>();
        if (!d->open(dit_path, s, err)) throw std::runtime_error("SeedVR2's DiT (" + dit_path + "): " + err);
        vae = std::move(v);
        dit = std::move(d);
    }
    template<class T> T* get(int64_t n) {
        T* p = nullptr;
        const uint64_t b = (uint64_t) std::max<int64_t>(n, 1) * sizeof(T);
        if (b > core::max_alloc_bytes()) throw std::runtime_error("SeedVR2: a buffer past the device's largest allocation");
        if (!gpu::alloc_device(&p, (size_t) b))
            throw std::runtime_error(std::string("SeedVR2: GPU memory: ") + gpu::last_error());
        scratch.push_back(p);
        scratch_bytes += b;
        return p;
    }
    void drop_scratch() {
        for (void* p : scratch) gpu::free(p);
        scratch.clear();
        scratch_bytes = 0;
    }
};

std::string Seedvr2::vae_path(const std::string& dit) {
    const size_t slash = dit.find_last_of("/\\");
    return (slash == std::string::npos ? std::string() : dit.substr(0, slash + 1)) + "ema_vae_fp16.safetensors";
}

Seedvr2::Seedvr2(const std::string& path) : impl_(std::make_unique<Impl>()) {
    Impl& m = *impl_;
    m.s = core::Runtime::get().create_stream();
    m.dit_path = path;
    m.vae_path = vae_path(path);
    {   // the sizes, read without the weights going to the GPU
        std::string err;
        image::MfVae v;
        if (!v.open(m.vae_path, m.s, err, true) || !v.open_encoder(err, true))
            throw std::runtime_error("SeedVR2's VAE (" + m.vae_path + "): " + err);
        m.vae_bytes = v.weight_bytes() + v.encoder_bytes();
        Seedvr2Dit d;
        if (!d.open(path, m.s, err, true)) throw std::runtime_error("SeedVR2's DiT (" + path + "): " + err);
        m.dit_bytes = d.weight_bytes() + d.block_bytes();
        m.dit_least = d.weight_bytes() + d.block_min();
    }
    info_.name = "seedvr2";
    info_.scale = 0;
    info_.multiple = 16;
    info_.frame_min = 1;
    info_.frame_step = 4;
    info_.context = 0;
    info_.color = ColorFix::lab;   // the reference pipeline's default correction
}

Seedvr2::~Seedvr2() {
    try {
        impl_->drop_scratch();
        core::Runtime::get().stream(impl_->s).wait();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-image: SeedVR2's queue at its end: %s\n", e.what());
    }
}

uint64_t Seedvr2::weight_bytes() const { return impl_->vae_bytes + impl_->dit_bytes; }

uint64_t Seedvr2::weight_least() const { return impl_->vae_bytes + impl_->dit_least; }

void Seedvr2::set_room(uint64_t bytes) { impl_->room = bytes; }

// a frame of w x h: its RGB, the latent's noise, condition, velocity and step (h / 16 x w / 16 tokens of 64), the
// decoder's planes and the larger of the VAE's and the DiT's working buffers
uint64_t Seedvr2::clip_bytes(int64_t w, int64_t h, int64_t frames) const {
    (void) frames;   // one frame at a time
    const int64_t th = h / 16, tw = w / 16, HW = w * h;
    const uint64_t own = (uint64_t) (3 * HW + 4 * th * tw * 64 + 4 * HW) * 4;
    const Impl& m = *impl_;
    if (!m.vae || !m.dit) {   // the parts' sizes from their files, read without their weights
        std::string err;
        image::MfVae v;
        Seedvr2Dit d;
        if (!v.open(m.vae_path, m.s, err, true) || !v.open_encoder(err, true) || !d.open(m.dit_path, m.s, err, true))
            throw std::runtime_error("SeedVR2: " + err);
        return own + std::max(std::max(v.encode_scratch_bytes(th, tw), v.scratch_bytes(th, tw)),
                              d.activation_bytes(th, tw));
    }
    const uint64_t vae = std::max(m.vae->encode_scratch_bytes(th, tw), m.vae->scratch_bytes(th, tw));
    return own + std::max(vae, m.dit->activation_bytes(th, tw));
}

uint64_t Seedvr2::clip_largest(int64_t w, int64_t h, int64_t frames) const {
    (void) frames;
    return (uint64_t) (4 * w * h) * 4 + (uint64_t) ((h / 16) * (w / 16)) * 12288 * 4;
}

void Seedvr2::run_clip(const uint8_t* in, int64_t w, int64_t h, int64_t frames, uint8_t* out, uint64_t seed) {
    if (w % 16 != 0 || h % 16 != 0) throw std::invalid_argument("SeedVR2: a frame's sides not multiples of 16");
    Impl& m = *impl_;
    m.load();
    {   // the blocks that stay.  Where every block and the encoder's pass whole fit the room, the blocks stay and the
        // decoder goes in tiles where it must (its front and middle whole, its last levels in tiles: they cost little);
        // else the VAE's passes take their room whole first and the blocks what is left (two slots at least), the
        // encoder's tiles (each the whole encoder over its part) costing more than the blocks' streaming
        const int64_t th0 = h / 16, tw0 = w / 16;
        const uint64_t own = (uint64_t) (3 * w * h + 4 * th0 * tw0 * 64 + 4 * w * h) * 4, margin = 512ull << 20;
        const uint64_t fixed = m.vae_bytes + m.dit->weight_bytes() + own + margin;
        const uint64_t act = m.dit->activation_bytes(th0, tw0), enc = m.vae->encode_scratch_bytes(th0, tw0);
        const uint64_t vae_least = std::max(m.vae->least_scratch_bytes(th0, tw0),
                                            m.vae->encode_scratch_bytes(std::min<int64_t>(th0, 16), std::min<int64_t>(tw0, 16)));
        const uint64_t vae = m.room >= fixed + m.dit->block_bytes() + std::max(act, enc)
                                 ? vae_least
                                 : std::max(m.vae->scratch_bytes(th0, tw0), enc);
        const uint64_t other = fixed + std::max(act, vae);
        m.dit->set_block_budget(std::max(m.room > other ? m.room - other : 0, m.dit->block_min()));
    }
    const int64_t th = h / 16, tw = w / 16, HW = w * h, n = th * tw * 64, lat = 16, lh = h / 8, lw = w / 8;
    float* rgb = m.get<float>(3 * HW);
    float* cond = m.get<float>(n);
    float* noise = m.get<float>(n);
    float* vel = m.get<float>(n);
    float* planes = m.get<float>(4 * HW);
    m.enc_s = m.dit_s = m.dec_s = 0.0;
    auto t0 = std::chrono::steady_clock::now();
    const auto lap = [&](double& acc) {   // the stream's work so far into acc
        core::Runtime::get().stream(m.s).wait_and_throw();
        const auto t1 = std::chrono::steady_clock::now();
        acc += std::chrono::duration<double>(t1 - t0).count();
        t0 = t1;
    };
    for (int64_t f = 0; f < frames; ++f) {
        const uint8_t* fin = in + f * HW * 4;
        rgb_of(fin, HW, rgb, m.s);
        {   // the encoder in the VRAM that is free now (in tiles where the whole does not fit), a margin kept
            size_t free = 0, total = 0;
            gpu::mem_info(&free, &total);
            const uint64_t margin = 256ull << 20;
            m.vae->encode(rgb, th, tw, nullptr, cond, free > margin ? (uint64_t) free - margin : 0);
        }
        scale(cond, n, kScaling, m.s);
        lap(m.enc_s);
        m.vae->drop_scratch();   // the three parts take the room in turn: each one's buffers go before the next's
        // the noise in the latent's [16][h][w] order (torch.randn's), to the tokens' (value 4 c + 2 dy + dx)
        const std::vector<float> r = image::torch_cpu_randn(seed + (uint64_t) f, (uint64_t) n);
        std::vector<float> tok((size_t) n);
        for (int64_t c = 0; c < lat; ++c)
            for (int64_t y = 0; y < lh; ++y)
                for (int64_t x = 0; x < lw; ++x)
                    tok[(size_t) (((y / 2) * tw + x / 2) * 64 + 4 * c + 2 * (y % 2) + x % 2)] =
                        r[(size_t) ((c * lh + y) * lw + x)];
        gpu::copy(noise, tok.data(), tok.size() * 4);
        m.dit->forward(noise, cond, th, tw, vel);
        lap(m.dit_s);
        m.dit->drop_activations();
        restore(noise, vel, n, 1.f / kScaling, cond, m.s);
        {   // the decoder in the VRAM that is free now (in tiles where the whole does not fit), a margin kept
            size_t free = 0, total = 0;
            gpu::mem_info(&free, &total);
            const uint64_t margin = 256ull << 20;
            m.vae->decode(cond, th, tw, planes, free > margin ? (uint64_t) free - margin : 0);
        }
        rgba8_of(planes, fin, HW, out + f * HW * 4, m.s);
        lap(m.dec_s);
    }
    core::Runtime::get().stream(m.s).wait_and_throw();
}

void Seedvr2::release(bool all) {
    Impl& m = *impl_;
    core::Runtime::get().stream(m.s).wait();
    m.drop_scratch();
    if (!m.vae || !m.dit) return;
    if (all) {   // the weights go with the parts: opened again on the next run
        m.dit.reset();
        m.vae.reset();
        return;
    }
    m.dit->drop_activations();
    m.vae->drop_scratch();
}

std::string Seedvr2::parts() const {
    char b[160];
    std::snprintf(b, sizeof b, "encode_s=%.3f dit_s=%.3f decode_s=%.3f blocks=%d", impl_->enc_s, impl_->dit_s,
                  impl_->dec_s, impl_->dit ? impl_->dit->resident_blocks() : 0);
    return b;
}

uint64_t Seedvr2::held() const {
    const Impl& m = *impl_;
    if (!m.vae || !m.dit) return m.scratch_bytes;
    return m.vae_bytes + m.dit->weight_bytes() + m.dit->blocks_vram() + m.dit->activations_held() +
           m.vae->scratch_held() + m.scratch_bytes;
}

}  // namespace strata::upscale
