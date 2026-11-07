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
#include <cmath>
#include <deque>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::upscale {

namespace seedvr2_detail {

sycl::queue& q_of(void* s) { return core::Runtime::get().stream(s); }

// A video's VAE in tiles of tt x tt tokens (0: the frame whole) over a th x tw latent of HW pixels: a slice's work,
// every tile's carried frames (one tile's where they are kept in host memory: `host`), the slice's frames and a tile's
uint64_t vae_video_need(const image::MfVae& vae, int64_t th, int64_t tw, int64_t HW, int64_t tt, bool host) {
    if (tt <= 0) tt = std::max(th, tw);
    const int64_t a = std::min(th, tt), b = std::min(tw, tt);
    // the tiles along a side of n: tt apart less the 4 tokens they share, the last flush with the edge
    const auto along = [](int64_t n, int64_t t) { return n <= t ? 1 : (n - 5) / (t - 4) + 1; };
    const auto tiles = host ? 1 : (uint64_t) (along(th, a) * along(tw, b));
    return vae.video_scratch_bytes(a, b, 5) + vae.video_carried_bytes(a, b) * tiles +
           (uint64_t) HW * 15 * 4 * 2;
}

// the tiles video_begin tries after the frame whole, largest first
constexpr int64_t kVaeTiles[] = {64, 48, 40, 32, 24, 16};

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

// a video's decoded frames rgb [T][3][HW] (about -1 .. 1) to RGBA8 [T][HW][4], the alpha 255
void rgba8_of_raw(const float* rgb, int64_t T, int64_t HW, uint8_t* out, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (T * HW)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t f = i / HW, p = i - f * HW;
        for (int c = 0; c < 3; ++c)
            out[i * 4 + c] =
                (uint8_t) sycl::round(sycl::clamp((rgb[(f * 3 + c) * HW + p] + 1.f) * 0.5f, 0.f, 1.f) * 255.f);
        out[i * 4 + 3] = 255;
    });
}

// a = a w + b (1 - w)
void blend(float* a, const float* b, int64_t n, float w, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = id[0];
        a[i] = a[i] * w + b[i] * (1.f - w);
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
    bool video_vae = false;   // the VAE opened in its video form
    int64_t vae_tt = 0;       // a video's VAE tiles (tokens a side; 0: the frame whole)
    bool vae_host = false;    // their carried frames in host memory (MfVae::set_video_host_cache)
    int64_t last_chunk = 0;   // the last video's chunk (frames), for parts()
    // a video's state (video_begin .. video_end)
    struct Video {
        bool on = false, ended = false;
        int64_t w = 0, h = 0, frames = 0, total = 0;   // the clip's frames (known at its end), padded to 4 n + 1
        int64_t cl = 1, tl = 1, overlap = 0, step = 1;  // a chunk's latents, the clip's, the overlap and the stride
        uint64_t seed = 0;
        int64_t pushed = 0, pend = 0, encoded = 0;      // frames pushed, waiting for the encoder, encoded
        int64_t latents = 0, cond0 = 0;                 // latents encoded; the first one `cond` holds
        std::deque<float*> cond;                        // the condition latents (scaled), [h w][64] each
        int64_t merged0 = 0, filled = 0;                // the restored latents from merged0, filled up to `filled`
        std::deque<float*> merged;
        int64_t next = 0, decoded = 0, out = 0;         // the next chunk; latents decoded; frames made
        std::deque<uint8_t*> ready;                     // frames ready, RGBA8
        float *slice = nullptr, *last = nullptr;        // the frames for the encoder [5][3][H][W]; the last frame's
        float *zbuf = nullptr, *ncat = nullptr, *ccat = nullptr, *vel = nullptr, *rgb = nullptr;
        uint8_t* rgba = nullptr;
        std::vector<float*> pool;                       // latents' buffers to use again
        std::vector<uint8_t*> fpool;                    // frames' buffers to use again
    } v;
    float* latent_buf() {
        if (!v.pool.empty()) {
            float* p = v.pool.back();
            v.pool.pop_back();
            return p;
        }
        return get<float>((v.h / 16) * (v.w / 16) * 64);
    }
    uint8_t* frame_buf() {
        if (!v.fpool.empty()) {
            uint8_t* p = v.fpool.back();
            v.fpool.pop_back();
            return p;
        }
        return get<uint8_t>(v.w * v.h * 4);
    }
    void load() {
        if (vae && dit) return;
        std::string err;
        auto v = std::make_unique<image::MfVae>();
        v->set_video(video_vae);
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
        v = Video();   // its buffers were the scratch's
    }
    void encode_slice();
    void run_chunks();
    void decode_upto(int64_t upto);
};

// the frames waiting for the encoder through it: the latents they give, scaled, to the conditions
void Seedvr2::Impl::encode_slice() {
    const int64_t th = v.h / 16, tw = v.w / 16, n = th * tw * 64;
    const int64_t k = vae->encode_video_tiled(v.slice, v.pend, th, tw, v.encoded == 0, v.zbuf, vae_tt);
    vae->drop_scratch();   // the room the DiT's chunk takes next (the carried frames stay)
    for (int64_t i = 0; i < k; ++i) {
        float* z = latent_buf();
        gpu::copy_async(z, v.zbuf + i * n, (size_t) n * 4, s);
        scale(z, n, kScaling, s);
        v.cond.push_back(z);
    }
    v.latents += k;
    v.encoded += v.pend;
    v.pend = 0;
}

// the chunks whose latents are all encoded, each through the DiT and merged; then the latents no later chunk touches
// decoded.  Before the clip's end only whole chunks; at it (v.ended, v.tl known) the rest, the last cut at the clip's
// end, none after the one that reaches it (SeedVR2TemporalChunk's)
void Seedvr2::Impl::run_chunks() {
    const int64_t th = v.h / 16, tw = v.w / 16, n = th * tw * 64, lh = v.h / 8, lw = v.w / 8;
    const int64_t end = v.ended ? v.tl : INT64_MAX;
    for (;;) {
        if (v.ended && v.filled >= v.tl) break;
        const int64_t s0 = v.next * v.step, e = std::min(s0 + v.cl, end), len = e - s0;
        if (s0 >= end || e > v.latents) break;
        // the chunk's conditions one after another, and its noise: the seed's for its shape [16][len][lh][lw]
        for (int64_t j = 0; j < len; ++j)
            gpu::copy_async(v.ccat + j * n, v.cond[(size_t) (s0 + j - v.cond0)], (size_t) n * 4, s);
        const std::vector<float> r = image::torch_cpu_randn(v.seed, (uint64_t) (16 * len * lh * lw));
        std::vector<float> tok((size_t) (len * n));
        for (int64_t c = 0; c < 16; ++c)
            for (int64_t f = 0; f < len; ++f)
                for (int64_t y = 0; y < lh; ++y)
                    for (int64_t x = 0; x < lw; ++x)
                        tok[(size_t) (f * n + ((y / 2) * tw + x / 2) * 64 + 4 * c + 2 * (y % 2) + x % 2)] =
                            r[(size_t) (((c * len + f) * lh + y) * lw + x)];
        gpu::copy(v.ncat, tok.data(), tok.size() * 4);
        auto t0 = std::chrono::steady_clock::now();
        dit->forward(v.ncat, v.ccat, th, tw, v.vel, len);
        core::Runtime::get().stream(s).wait_and_throw();
        dit->drop_activations();   // the room the VAE's slices take
        dit_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        restore(v.ncat, v.vel, len * n, 1.f / kScaling, v.vel, s);
        // merged: the overlap crossfaded (ComfyUI's SeedVR2TemporalMerge), the rest appended
        const int64_t fade = std::min(v.filled - s0, len);
        for (int64_t j = 0; j < len; ++j) {
            const int64_t L = s0 + j;
            if (j < fade) {
                float ramp = fade > 1 ? (float) j / (float) (fade - 1) : 0.f;
                ramp = std::clamp((ramp - 1.f / 3.f) / (1.f / 3.f), 0.f, 1.f);
                const float wprev = 0.5f + 0.5f * std::cos(3.14159265358979f * ramp);
                blend(v.merged[(size_t) (L - v.merged0)], v.vel + j * n, n, wprev, s);
            } else {
                float* z = latent_buf();
                gpu::copy_async(z, v.vel + j * n, (size_t) n * 4, s);
                v.merged.push_back(z);
            }
        }
        v.filled = e;
        ++v.next;
        // the conditions no later chunk reads go
        const int64_t keep = std::min(v.next * v.step, end);
        while (v.cond0 < keep && !v.cond.empty()) {
            v.pool.push_back(v.cond.front());
            v.cond.pop_front();
            ++v.cond0;
        }
        decode_upto(v.ended && e >= v.tl ? v.tl : keep);
    }
}

// the restored latents up to `upto` through the decoder, one at a time (the first gives one frame, each next four):
// their frames ready, the padding's dropped
void Seedvr2::Impl::decode_upto(int64_t upto) {
    const int64_t th = v.h / 16, tw = v.w / 16, HW = v.w * v.h;
    upto = std::min(upto, v.filled);
    while (v.decoded < upto) {
        auto t0 = std::chrono::steady_clock::now();
        const int64_t T = vae->decode_video_tiled(v.merged[(size_t) (v.decoded - v.merged0)], 1, th, tw,
                                                  v.decoded == 0, v.rgb, vae_tt);
        vae->drop_scratch();
        rgba8_of_raw(v.rgb, T, HW, v.rgba, s);
        for (int64_t f = 0; f < T && (!v.ended || v.out < v.frames); ++f, ++v.out) {
            uint8_t* fr = frame_buf();
            gpu::copy_async(fr, v.rgba + f * HW * 4, (size_t) HW * 4, s);
            v.ready.push_back(fr);
        }
        core::Runtime::get().stream(s).wait_and_throw();
        dec_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        ++v.decoded;
        v.pool.push_back(v.merged.front());   // the decoded latent goes
        v.merged.pop_front();
        ++v.merged0;
    }
}

uint64_t Seedvr2::video_bytes(int64_t w, int64_t h, int64_t chunk, bool least) const {
    const int64_t th = h / 16, tw = w / 16, n = th * tw * 64, cl = (std::max<int64_t>(chunk, 1) - 1) / 4 + 1;
    const Impl& m = *impl_;
    std::string err;
    image::MfVae v;
    v.set_video(true);
    Seedvr2Dit d;
    if (!v.open(m.vae_path, m.s, err, true) || !v.open_encoder(err, true) || !d.open(m.dit_path, m.s, err, true))
        throw std::runtime_error("SeedVR2: " + err);
    // as video_begin takes the room besides the weights: the chunk's noise, conditions, velocity and latents held,
    // the frames, then the larger of the chunk's DiT activations and the VAE's slice (the frame whole for the full;
    // for the least, the smallest tiles with their carried frames in host memory)
    const int64_t HW = w * h;
    uint64_t vae = vae_video_need(v, th, tw, HW, 0, false);
    if (least)   // the smallest tiles, their carried frames in host memory
        vae = std::min(vae, vae_video_need(v, th, tw, HW, std::end(kVaeTiles)[-1], true));
    return (uint64_t) (cl * n) * 4 * 7 + (uint64_t) HW * 4 * 27 + std::max(d.activation_bytes(th, tw, cl), vae);
}

void Seedvr2::video_begin(int64_t w, int64_t h, int64_t frames, uint64_t seed, int64_t chunk, int64_t overlap) {
    if (w % 16 != 0 || h % 16 != 0) throw std::invalid_argument("SeedVR2: a frame's sides not multiples of 16");
    if (frames < 1) throw std::invalid_argument("SeedVR2: a video of no frames");
    Impl& m = *impl_;
    if (!m.video_vae) {   // the VAE again in its video form
        m.video_vae = true;
        m.vae.reset();
        m.dit.reset();
        m.vae_bytes = 0;
    }
    m.load();
    if (m.vae_bytes == 0 || !m.vae->video()) m.vae_bytes = m.vae->weight_bytes() + m.vae->encoder_bytes();
    m.drop_scratch();
    m.vae->video_reset();
    Impl::Video& v = m.v;
    v.on = true;
    v.w = w;
    v.h = h;
    v.frames = frames;   // a hint until the end (the chunk's largest; the reader's count may differ)
    v.total = (frames - 1 + 3) / 4 * 4 + 1;
    v.tl = (v.total - 1) / 4 + 1;
    const int64_t th = h / 16, tw = w / 16, n = th * tw * 64, HW = w * h;
    // The room: the weights (the DiT's blocks at their least), the latents held and the frames, then the larger of a
    // chunk's DiT activations and the VAE's slice (they take the room in turns), the rest the DiT's resident blocks.
    // The chunk (0: the largest that fits) and the VAE's tiles (the largest that fit: the frame whole where it does)
    // from it
    const uint64_t margin = 512ull << 20;
    const auto fixed = [&](int64_t cl) {
        return m.vae_bytes + m.dit->weight_bytes() + m.dit->block_min() + (uint64_t) (cl * n) * 4 * 7 +
               (uint64_t) HW * 4 * 27 + margin;
    };
    const auto work = [&](int64_t cl) {
        const uint64_t f = fixed(cl);
        return m.room > f ? m.room - f : 0;
    };
    int64_t cl = chunk > 0 ? (chunk - 1) / 4 + 1 : v.tl;
    if (chunk > 0 && (chunk - 1) % 4 != 0) throw std::invalid_argument("SeedVR2: a chunk of 4 n + 1 frames");
    cl = std::min(cl, v.tl);
    if (chunk <= 0)
        while (cl > 1 && m.dit->activation_bytes(th, tw, cl) > work(cl)) --cl;
    v.cl = cl;
    m.last_chunk = 4 * (cl - 1) + 1;
    v.overlap = std::clamp<int64_t>(overlap, 0, v.cl - 1);
    v.step = v.cl - v.overlap;
    v.seed = seed;
    const uint64_t act = m.dit->activation_bytes(th, tw, v.cl), room = work(v.cl);
    {   // the tiles: the frame whole, else the largest of 64 .. 16 tokens whose slice and carried frames fit, else the
        // largest whose slice and one tile's carried frames fit, the tiles' carried frames in host memory (copied in
        // and out each slice: slower, but the GPU holds one tile's)
        const auto need = [&](int64_t tt, bool host) { return vae_video_need(*m.vae, th, tw, HW, tt, host); };
        m.vae_tt = 0;
        m.vae_host = false;
        if (need(0, false) > room) {
            const auto pick = [&](bool host) {
                for (const int64_t tt : kVaeTiles)
                    if (tt < std::max(th, tw) && need(tt, host) <= room) return tt;
                return int64_t{0};
            };
            m.vae_tt = pick(false);
            if (m.vae_tt == 0) {
                m.vae_tt = pick(true);
                m.vae_host = true;
                if (m.vae_tt == 0) m.vae_tt = std::end(kVaeTiles)[-1];   // the least, whatever the room says
            }
        }
        m.vae->set_video_host_cache(m.vae_host);
        const uint64_t vae = need(m.vae_tt, m.vae_host);
        const uint64_t f = fixed(v.cl) - m.dit->block_min() + std::max(act, vae);
        m.dit->set_block_budget(std::max(m.room > f ? m.room - f : 0, m.dit->block_min()));
    }
    v.slice = m.get<float>(HW * 15);
    v.last = m.get<float>(3 * HW);
    v.zbuf = m.get<float>(2 * n);
    v.ncat = m.get<float>(v.cl * n);
    v.ccat = m.get<float>(v.cl * n);
    v.vel = m.get<float>(v.cl * n);
    v.rgb = m.get<float>(HW * 12);
    v.rgba = m.get<uint8_t>(4 * HW * 4);
    m.enc_s = m.dit_s = m.dec_s = 0.0;
}

void Seedvr2::video_push(const uint8_t* in, int64_t n) {
    Impl& m = *impl_;
    Impl::Video& v = m.v;
    if (!v.on) throw std::logic_error("SeedVR2: a video's frames before video_begin");
    const int64_t HW = v.w * v.h;
    for (int64_t f = 0; f < n; ++f) {
        rgb_of(in + f * HW * 4, HW, v.slice + v.pend * 3 * HW, m.s);
        ++v.pend;
        ++v.pushed;
        if (v.pend == (v.encoded == 0 ? 5 : 4)) {   // the encoder's slice: 1 + 4 frames first, 4 after
            gpu::copy_async(v.last, v.slice + (v.pend - 1) * 3 * HW, (size_t) (3 * HW) * 4, m.s);
            auto t0 = std::chrono::steady_clock::now();
            m.encode_slice();
            m.enc_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            m.run_chunks();
        }
    }
}

int64_t Seedvr2::video_ready() const { return (int64_t) impl_->v.ready.size(); }

void Seedvr2::video_take(uint8_t* out, int64_t n) {
    Impl::Video& v = impl_->v;
    if (n > (int64_t) v.ready.size()) throw std::invalid_argument("SeedVR2: fewer frames ready than asked for");
    const int64_t fb = v.w * v.h * 4;
    for (int64_t f = 0; f < n; ++f) {
        gpu::copy_async(out + f * fb, v.ready.front(), (size_t) fb, impl_->s);
        v.fpool.push_back(v.ready.front());
        v.ready.pop_front();
    }
    core::Runtime::get().stream(impl_->s).wait_and_throw();
}

void Seedvr2::video_end() {
    Impl& m = *impl_;
    Impl::Video& v = m.v;
    if (!v.on) throw std::logic_error("SeedVR2: video_end before video_begin");
    if (v.pushed < 1) throw std::invalid_argument("SeedVR2: a video of no frames");
    const int64_t HW = v.w * v.h;
    // the clip as it came: its frames padded to 4 n + 1 by the last one
    v.ended = true;
    v.frames = v.pushed;
    v.total = (v.frames - 1 + 3) / 4 * 4 + 1;
    v.tl = (v.total - 1) / 4 + 1;
    if (v.pend > 0) gpu::copy_async(v.last, v.slice + (v.pend - 1) * 3 * HW, (size_t) (3 * HW) * 4, m.s);
    while (v.encoded + v.pend < v.total) {
        gpu::copy_async(v.slice + v.pend * 3 * HW, v.last, (size_t) (3 * HW) * 4, m.s);
        ++v.pend;
        if (v.pend == (v.encoded == 0 ? std::min<int64_t>(5, v.total) : 4)) {
            auto t0 = std::chrono::steady_clock::now();
            m.encode_slice();
            m.enc_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
    }
    if (v.pend > 0) {   // a first slice of one frame (a clip of one)
        auto t0 = std::chrono::steady_clock::now();
        m.encode_slice();
        m.enc_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    m.run_chunks();
    m.decode_upto(v.tl);
    v.on = false;
}

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
    char b[220];
    std::snprintf(b, sizeof b, "encode_s=%.3f dit_s=%.3f decode_s=%.3f blocks=%d chunk=%lld vae_tile=%lld vae_host=%d",
                  impl_->enc_s, impl_->dit_s, impl_->dec_s, impl_->dit ? impl_->dit->resident_blocks() : 0,
                  (long long) impl_->last_chunk, (long long) impl_->vae_tt, (int) impl_->vae_host);
    return b;
}

uint64_t Seedvr2::held() const {
    const Impl& m = *impl_;
    if (!m.vae || !m.dit) return m.scratch_bytes;
    return m.vae_bytes + m.dit->weight_bytes() + m.dit->blocks_vram() + m.dit->activations_held() +
           m.vae->scratch_held() + m.scratch_bytes;
}

}  // namespace strata::upscale
