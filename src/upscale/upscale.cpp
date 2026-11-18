// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/upscale/upscale.cpp - see include/strata/upscale/upscale.hpp.

#include "strata/upscale/upscale.hpp"

#include "strata/core/device.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/img_color.hpp"
#include "strata/image/img_crop.hpp"
#include "strata/upscale/esrgan.hpp"
#include "strata/upscale/seedvr2.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <deque>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strata::upscale {

namespace {

sycl::queue& queue() { return core::Runtime::get().stream(nullptr); }

void done(const sycl::event& e, const char* what) { core::Runtime::get().wait(e, what); }

double since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// a device buffer, freed with it
struct Buf {
    uint8_t* p = nullptr;
    Buf() = default;
    explicit Buf(uint64_t bytes) {
        if (bytes > core::max_alloc_bytes())
            throw std::runtime_error("upscale: a picture past the device's largest allocation");
        if (!gpu::alloc_device(&p, (size_t) bytes))
            throw std::runtime_error(std::string("upscale: GPU memory: ") + gpu::last_error());
    }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    Buf& operator=(Buf&& o) noexcept {
        std::swap(p, o.p);
        return *this;
    }
    ~Buf() {
        if (p != nullptr) gpu::free(p);
    }
};

// Lanczos: the driver's resize alone
class Lanczos final : public UpscaleModel {
public:
    Lanczos() {
        info_.name = "lanczos";
        info_.identity = true;
    }
    const ModelInfo& info() const override { return info_; }
    uint64_t weight_bytes() const override { return 0; }
    uint64_t clip_bytes(int64_t, int64_t, int64_t) const override { return 0; }
    uint64_t clip_largest(int64_t, int64_t, int64_t) const override { return 0; }
    void run_clip(const uint8_t* in, int64_t w, int64_t h, int64_t frames, uint8_t* out, uint64_t) override {
        if (in != out) gpu::copy(out, in, (size_t) (frames * w * h * 4));
    }
    void release(bool) override {}
    uint64_t held() const override { return 0; }

private:
    ModelInfo info_;
};

// the frames a clip of `n` takes: frame_min + k * frame_step, at least n
int64_t clip_frames(const ModelInfo& in, int64_t n) {
    if (n <= in.frame_min) return in.frame_min;
    const int64_t step = std::max<int64_t>(1, in.frame_step);
    return in.frame_min + (n - in.frame_min + step - 1) / step * step;
}

int64_t padded(int64_t side, int64_t multiple) { return (side + multiple - 1) / multiple * multiple; }

// what lives through the whole run beside the model's work: the input, the model's output at its size and the output
// at the size asked for (where they differ), and the correction's reference and scratch
uint64_t frame_bytes(const ModelInfo& in, int64_t sw, int64_t sh, int64_t w, int64_t h, ColorFix color) {
    const int64_t s = in.scale > 0 ? in.scale : 1;
    const int64_t mw = in.scale > 0 ? sw * s : padded(w, in.multiple), mh = in.scale > 0 ? sh * s : padded(h, in.multiple);
    uint64_t b = (uint64_t) (sw * sh * 4) + (uint64_t) (mw * mh * 4) * (uint64_t) clip_frames(in, 1);
    if (in.scale > 0 ? (mw != w || mh != h) : true) b += (uint64_t) (w * h * 4);
    // the resize's scratch (once at a time): the model's output to w x h, or the input to it
    uint64_t post = in.scale > 0 ? image::lanczos_resize_bytes(mw, mh, w, h) : image::lanczos_resize_bytes(sw, sh, w, h);
    if (color == ColorFix::wavelet || color == ColorFix::lab)
        post = std::max(post, (uint64_t) (w * h * 4) + std::max(image::lanczos_resize_bytes(sw, sh, w, h),
                                                                   color == ColorFix::lab ? image::color::lab_match_bytes(w, h)
                                                                                          : image::color::wavelet_bytes(w, h)));
    return b + post;
}

// the window of a tile: its core of at most `t` and the context around it, within the side
int64_t window(int64_t t, int64_t context, int64_t side) { return std::min(side, t + 2 * context); }

}  // namespace

ColorFix parse_color(const std::string& word) {
    if (word == "none") return ColorFix::none;
    if (word == "lab") return ColorFix::lab;
    if (word == "adain") return ColorFix::adain;
    if (word == "wavelet") return ColorFix::wavelet;
    throw std::invalid_argument("upscale: a color correction of none, lab, adain or wavelet, not " + word);
}

const char* color_name(ColorFix c) {
    switch (c) {
    case ColorFix::lab: return "lab";
    case ColorFix::adain: return "adain";
    case ColorFix::wavelet: return "wavelet";
    case ColorFix::none: break;
    }
    return "none";
}

std::unique_ptr<UpscaleModel> open_model(const std::string& name, const std::string& path) {
    if (name == "lanczos") return std::make_unique<Lanczos>();
    if (name == "realesrgan") {
        if (path.empty()) throw std::invalid_argument("upscale: realesrgan needs its weights' file");
        return std::make_unique<Esrgan>(path);
    }
    if (name == "seedvr2") {
        if (path.empty()) throw std::invalid_argument("upscale: seedvr2 needs its DiT's file");
        return std::make_unique<Seedvr2>(path);
    }
    throw std::invalid_argument("upscale: no upscaler " + name + " (lanczos, realesrgan, seedvr2)");
}

uint64_t UpscaleRunner::least_bytes(const UpscaleModel& m, int64_t sw, int64_t sh, int64_t w, int64_t h,
                                    ColorFix color) {
    const ModelInfo& in = m.info();
    uint64_t work = 0;
    if (!in.identity) {
        const int64_t fw = in.scale > 0 ? sw : padded(w, in.multiple), fh = in.scale > 0 ? sh : padded(h, in.multiple);
        const int64_t t = in.context > 0 ? in.min_tile : std::max(fw, fh);
        const int64_t ww = window(t, in.context, fw), wh = window(t, in.context, fh), s = std::max(1, in.scale);
        const uint64_t tiles = in.context > 0 && (ww < fw || wh < fh) ? (uint64_t) (ww * wh) * 4 * (1 + s * s) : 0;
        work = m.weight_least() + m.clip_least(ww, wh, clip_frames(in, 1)) + tiles;
    }
    return frame_bytes(in, sw, sh, w, h, color) + work;
}

uint64_t UpscaleRunner::full_bytes(const UpscaleModel& m, int64_t sw, int64_t sh, int64_t w, int64_t h,
                                   ColorFix color) {
    const ModelInfo& in = m.info();
    uint64_t work = 0;
    if (!in.identity) {
        const int64_t fw = in.scale > 0 ? sw : padded(w, in.multiple), fh = in.scale > 0 ? sh : padded(h, in.multiple);
        work = m.weight_bytes() + m.clip_bytes(fw, fh, clip_frames(in, 1));
    }
    return frame_bytes(in, sw, sh, w, h, color) + work;
}

QiRef UpscaleRunner::run(UpscaleModel& m, const QiRef& src, const Request& r, uint64_t avail, Report* rep,
                         const std::function<void(double)>& step) {
    const ModelInfo& in = m.info();
    const int64_t sw = src.width, sh = src.height, w = r.width, h = r.height;
    if (sw <= 0 || sh <= 0 || src.rgba.size() != (size_t) (sw * sh * 4))
        throw std::invalid_argument("upscale: an empty image");
    if (w <= 0 || h <= 0) throw std::invalid_argument("upscale: no size to upscale to");
    Report local;
    Report& rp = rep != nullptr ? *rep : local;
    rp = Report{};
    rp.color = r.color_set ? r.color : in.color;
    const int64_t nfr = clip_frames(in, 1);
    Buf pic((uint64_t) (sw * sh * 4));
    gpu::copy(pic.p, src.rgba.data(), src.rgba.size());
    // the model's input: the picture as it is (a native scale), or the picture at the size asked for, padded to the
    // model's multiple by repeating its edges (the padding cut off again after)
    const int s = in.scale > 0 ? in.scale : 1;
    const int64_t fw = in.scale > 0 ? sw : padded(w, in.multiple), fh = in.scale > 0 ? sh : padded(h, in.multiple);
    const int64_t mw = fw * s, mh = fh * s;   // the model's output
    Buf feed, made((uint64_t) (mw * mh * 4) * (uint64_t) nfr);
    const uint8_t* fin = pic.p;
    auto t0 = std::chrono::steady_clock::now();
    if (in.scale == 0) {
        feed = Buf((uint64_t) (fw * fh * 4) * (uint64_t) nfr);
        if (fw == w && fh == h) {
            image::lanczos_resize_device(pic.p, sw, sh, feed.p, w, h);
        } else {
            Buf sized((uint64_t) (w * h * 4));
            image::lanczos_resize_device(pic.p, sw, sh, sized.p, w, h);
            uint8_t* f = feed.p;
            const uint8_t* z = sized.p;
            done(queue().parallel_for(sycl::range<2>((size_t) fh, (size_t) fw), [=](sycl::id<2> id) {
                const int64_t y = (int64_t) id[0], x = (int64_t) id[1];
                const int64_t yy = sycl::min(y, h - 1), xx = sycl::min(x, w - 1);
                for (int c = 0; c < 4; ++c) f[(y * fw + x) * 4 + c] = z[(yy * w + xx) * 4 + c];
            }), "upscale: padding");
        }
        for (int64_t k = 1; k < nfr; ++k)   // the frame repeated up to the model's least clip
            gpu::copy(feed.p + k * fw * fh * 4, feed.p, (size_t) (fw * fh * 4));
        fin = feed.p;
        rp.resize_s += since(t0);
    }
    // the model, the frame whole or in tiles of the largest core whose run fits the VRAM left.  These are not the
    // VAEs' tile parts (src/image/tile_sides.hpp): a window here is a core plus the context the model needs around it,
    // cut out of the frame and its core pasted back with nothing overlapping - every output pixel comes from one
    // window, so there is no shared band to ramp, no weights to add up and no divide to end.  Its side too comes from
    // halving until the run fits, not from weighing the tiles' positions against each other
    t0 = std::chrono::steady_clock::now();
    // the share done (step): the tiles done, and the model's own share of the tile it is on
    int64_t ntiles = 1, tdone = 0;
    struct Unset {
        UpscaleModel& m;
        ~Unset() { m.progress = nullptr; }
    } unset{m};
    if (step) m.progress = [&](double f) { step(((double) tdone + std::clamp(f, 0.0, 1.0)) / (double) ntiles); };
    if (in.identity) {
        m.run_clip(fin, fw, fh, nfr, made.p, r.seed);
    } else {
        const uint64_t frame = frame_bytes(in, sw, sh, w, h, rp.color), fixed = frame + m.weight_least();
        const uint64_t room = avail > fixed ? avail - fixed : 0;
        m.set_room(avail > frame ? avail - frame : 0);
        int64_t T = std::max(fw, fh);
        if (r.max_tile > 0) T = std::min(T, r.max_tile);
        if (in.context > 0) {
            const auto fits = [&](int64_t t) {
                const int64_t ww = window(t, in.context, fw), wh = window(t, in.context, fh);
                const bool tiled = ww < fw || wh < fh;
                return m.clip_bytes(ww, wh, nfr) + (tiled ? (uint64_t) (ww * wh) * 4 * (1 + s * s) : 0) <= room &&
                       m.clip_largest(ww, wh, nfr) <= core::max_alloc_bytes();
            };
            while (T > in.min_tile && !fits(T)) T = std::max(in.min_tile, T / 2);
        }
        for (;;) {
            try {
                const int64_t nx = in.context > 0 ? (fw + T - 1) / T : 1, ny = in.context > 0 ? (fh + T - 1) / T : 1;
                ntiles = nx * ny;
                tdone = 0;
                if (nx * ny == 1) {
                    m.run_clip(fin, fw, fh, nfr, made.p, r.seed);
                    rp.tile = 0;
                    rp.tiles = 0;
                } else {
                    // the cores as even as the side allows; each window the core and the context around it, moved
                    // inside the frame, so that every tile has one shape
                    const int64_t cw = (fw + nx - 1) / nx, ch = (fh + ny - 1) / ny;
                    const int64_t ww = window(cw, in.context, fw), wh = window(ch, in.context, fh);
                    Buf tin((uint64_t) (ww * wh * 4) * (uint64_t) nfr), tout((uint64_t) (ww * wh * s * s * 4) * (uint64_t) nfr);
                    int64_t count = 0;
                    for (int64_t ty = 0; ty < ny; ++ty)
                        for (int64_t tx = 0; tx < nx; ++tx) {
                            const int64_t c0x = tx * cw, c1x = std::min(fw, c0x + cw);
                            const int64_t c0y = ty * ch, c1y = std::min(fh, c0y + ch);
                            if (c0x >= c1x || c0y >= c1y) continue;
                            const int64_t x0 = std::clamp<int64_t>(c0x - in.context, 0, fw - ww);
                            const int64_t y0 = std::clamp<int64_t>(c0y - in.context, 0, fh - wh);
                            uint8_t* ti = tin.p;
                            uint8_t* to = tout.p;
                            uint8_t* mo = made.p;
                            const int64_t frames = nfr;
                            done(queue().parallel_for(sycl::range<3>((size_t) frames, (size_t) wh, (size_t) ww),
                                                      [=](sycl::id<3> id) {
                                const int64_t f = (int64_t) id[0], y = (int64_t) id[1], x = (int64_t) id[2];
                                const uint8_t* a = fin + ((f * fh + y0 + y) * fw + x0 + x) * 4;
                                uint8_t* b = ti + ((f * wh + y) * ww + x) * 4;
                                for (int c = 0; c < 4; ++c) b[c] = a[c];
                            }), "upscale: tile");
                            m.run_clip(tin.p, ww, wh, nfr, tout.p, r.seed);
                            // the core's part of the output, into the model's output at its place
                            const int64_t ow = ww * s, oh = wh * s, ox = (c0x - x0) * s, oy = (c0y - y0) * s;
                            const int64_t kw = (c1x - c0x) * s, kh = (c1y - c0y) * s, px = c0x * s, py = c0y * s;
                            done(queue().parallel_for(sycl::range<3>((size_t) frames, (size_t) kh, (size_t) kw),
                                                      [=](sycl::id<3> id) {
                                const int64_t f = (int64_t) id[0], y = (int64_t) id[1], x = (int64_t) id[2];
                                const uint8_t* a = to + ((f * oh + oy + y) * ow + ox + x) * 4;
                                uint8_t* b = mo + ((f * mh + py + y) * mw + px + x) * 4;
                                for (int c = 0; c < 4; ++c) b[c] = a[c];
                            }), "upscale: tile out");
                            ++count;
                            tdone = count;
                            if (step) step((double) tdone / (double) ntiles);
                        }
                    rp.tile = T;
                    rp.tiles = count;
                }
                break;
            } catch (const std::runtime_error&) {   // no room for this tile: a smaller one
                if (in.context <= 0 || T <= in.min_tile) {
                    m.release(false);
                    throw;
                }
                m.release(false);
                T = std::max(in.min_tile, T / 2);
            }
        }
        m.release(false);
    }
    if (step) step(1.0);
    rp.model_s += since(t0);
    // the first frame at the size asked for
    t0 = std::chrono::steady_clock::now();
    Buf sized;
    uint8_t* out = made.p;
    if (in.scale > 0 && (mw != w || mh != h)) {
        sized = Buf((uint64_t) (w * h * 4));
        image::lanczos_resize_device(made.p, mw, mh, sized.p, w, h);
        out = sized.p;
    } else if (mw != w || mh != h) {   // the padding cut off
        sized = Buf((uint64_t) (w * h * 4));
        uint8_t* z = sized.p;
        const uint8_t* a = made.p;
        done(queue().parallel_for(sycl::range<2>((size_t) h, (size_t) w), [=](sycl::id<2> id) {
            const int64_t y = (int64_t) id[0], x = (int64_t) id[1];
            for (int c = 0; c < 4; ++c) z[(y * w + x) * 4 + c] = a[(y * mw + x) * 4 + c];
        }), "upscale: crop");
        out = sized.p;
    }
    rp.resize_s += since(t0);
    // the colors moved back toward the input's
    t0 = std::chrono::steady_clock::now();
    namespace color = image::color;
    if (rp.color == ColorFix::adain) {
        const color::Stats from = color::stats(out, w * h, color::Space::rgb),
                           to = color::stats(pic.p, sw * sh, color::Space::rgb);
        rp.color_applied = color::transfer(out, w * h, color::Space::rgb, from, to, 1.0);
    } else if (rp.color == ColorFix::wavelet || rp.color == ColorFix::lab) {   // against the input at the output's size
        Buf ref((uint64_t) (w * h * 4));
        image::lanczos_resize_device(pic.p, sw, sh, ref.p, w, h);
        if (rp.color == ColorFix::lab) color::lab_match(out, ref.p, w, h);
        else color::wavelet(out, ref.p, w, h);
        rp.color_applied = true;
    }
    rp.color_s += since(t0);
    QiRef res;
    res.width = w;
    res.height = h;
    res.rgba.resize((size_t) (w * h * 4));
    gpu::copy(res.rgba.data(), out, res.rgba.size());
    return res;
}

namespace {

// RGB24 rows to RGBA8 (the alpha 255), and back
void rgba_of_rgb(const uint8_t* rgb, int64_t n, std::vector<uint8_t>& rgba) {
    rgba.resize((size_t) (n * 4));
    for (int64_t p = 0; p < n; ++p) {
        rgba[(size_t) (p * 4)] = rgb[p * 3];
        rgba[(size_t) (p * 4 + 1)] = rgb[p * 3 + 1];
        rgba[(size_t) (p * 4 + 2)] = rgb[p * 3 + 2];
        rgba[(size_t) (p * 4 + 3)] = 255;
    }
}

void rgb_of_rgba(const std::vector<uint8_t>& rgba, uint8_t* rgb) {
    const size_t n = rgba.size() / 4;
    for (size_t p = 0; p < n; ++p)
        for (int c = 0; c < 3; ++c) rgb[p * 3 + c] = rgba[p * 4 + c];
}

}  // namespace

VideoReport UpscaleRunner::run_video(UpscaleModel& m, const std::string& in, const std::string& stem,
                                     const VideoRequest& r, uint64_t avail,
                                     const std::function<bool(int64_t, int64_t)>& step) {
    const ModelInfo& mi = m.info();
    const video::Info info = video::probe(in);
    const int64_t sw = info.width, sh = info.height, w = r.frame.width, h = r.frame.height;
    if (w <= 0 || h <= 0) throw std::invalid_argument("upscale: no size to upscale to");
    VideoReport rep;
    rep.width = w;
    rep.height = h;
    rep.fps = info.fps;
    rep.vfr = info.vfr;
    rep.color = r.frame.color_set ? r.frame.color : mi.color;
    video::WriteOptions wo;
    wo.codec = r.codec;
    wo.audio_from = in;
    video::Reader rd(in, info);
    video::Writer wr(stem, w, h, info.fps, wo);
    const int64_t fb = sw * sh * 3, ob = w * h * 3;
    constexpr int64_t kRead = 4;   // the frames read at a time
    std::vector<uint8_t> buf((size_t) (fb * kRead)), rgba, out_rgb((size_t) ob);
    const auto tell = [&](int64_t done) {
        if (step && !step(done, std::max(info.frames, done)))
            throw std::runtime_error("upscale: the video was stopped");
    };
    if (!m.temporal()) {   // each frame on its own, as an image
        for (;;) {
            const auto r0 = std::chrono::steady_clock::now();
            const int64_t n = rd.read(buf.data(), kRead);
            rep.read_s += since(r0);
            if (n <= 0) break;
            for (int64_t f = 0; f < n; ++f) {
                QiRef src;
                src.width = sw;
                src.height = sh;
                rgba_of_rgb(buf.data() + f * fb, sw * sh, src.rgba);
                Report fr;
                const QiRef big = run(m, src, r.frame, avail, &fr);
                rep.model_s += fr.model_s;
                rep.resize_s += fr.resize_s;
                rep.color_s += fr.color_s;
                rgb_of_rgba(big.rgba, out_rgb.data());
                const auto t0 = std::chrono::steady_clock::now();
                wr.write(out_rgb.data(), 1);
                rep.write_s += since(t0);
                tell(++rep.frames);
            }
            if (r.pause) r.pause();
        }
    } else {
        // the model's frames: the frame at the size asked for, padded to the model's multiple by its edges
        const int64_t fw = padded(w, mi.multiple), fh = padded(h, mi.multiple);
        Buf pic((uint64_t) (sw * sh * 4)), sized((uint64_t) (w * h * 4)), feed((uint64_t) (fw * fh * 4)),
            made((uint64_t) (fw * fh * 4)), out((uint64_t) (w * h * 4)), ref((uint64_t) (w * h * 4));
        const uint64_t frame = frame_bytes(mi, sw, sh, w, h, rep.color) + (uint64_t) (fw * fh * 4) * 2;
        m.set_room(avail > frame ? avail - frame : 0);
        try {
        m.video_begin(fw, fh, info.frames, r.frame.seed, r.chunk, r.overlap);
        std::deque<std::vector<uint8_t>> waiting;   // the frames pushed whose output is not out yet (their colors')
        std::vector<uint8_t> host((size_t) (w * h * 4));
        const auto drain = [&] {
            if (m.video_ready() <= 0) return;
            while (m.video_ready() > 0) {
                auto t0 = std::chrono::steady_clock::now();
                m.video_take(made.p, 1);
                rep.model_s += since(t0);
                t0 = std::chrono::steady_clock::now();
                uint8_t* z = out.p;
                const uint8_t* a = made.p;
                done(queue().parallel_for(sycl::range<2>((size_t) h, (size_t) w), [=](sycl::id<2> id) {
                    const int64_t y = (int64_t) id[0], x = (int64_t) id[1];
                    for (int c = 0; c < 4; ++c) z[(y * w + x) * 4 + c] = a[(y * fw + x) * 4 + c];
                }), "upscale: crop");
                rep.resize_s += since(t0);
                t0 = std::chrono::steady_clock::now();
                namespace color = image::color;
                if (rep.color != ColorFix::none) {   // against the frame it came from
                    rgba_of_rgb(waiting.front().data(), sw * sh, rgba);
                    gpu::copy(pic.p, rgba.data(), rgba.size());
                    if (rep.color == ColorFix::adain) {
                        const color::Stats from = color::stats(out.p, w * h, color::Space::rgb),
                                           to = color::stats(pic.p, sw * sh, color::Space::rgb);
                        color::transfer(out.p, w * h, color::Space::rgb, from, to, 1.0);
                    } else {
                        image::lanczos_resize_device(pic.p, sw, sh, ref.p, w, h);
                        if (rep.color == ColorFix::lab) color::lab_match(out.p, ref.p, w, h);
                        else color::wavelet(out.p, ref.p, w, h);
                    }
                }
                rep.color_s += since(t0);
                waiting.pop_front();
                gpu::copy(host.data(), out.p, host.size());
                rgb_of_rgba(host, out_rgb.data());
                t0 = std::chrono::steady_clock::now();
                wr.write(out_rgb.data(), 1);
                rep.write_s += since(t0);
                tell(++rep.frames);
            }
            if (r.pause) r.pause();   // a chunk's frames out
        };
        for (;;) {
            auto t0 = std::chrono::steady_clock::now();
            const int64_t n = rd.read(buf.data(), kRead);
            rep.read_s += since(t0);
            if (n <= 0) break;
            for (int64_t f = 0; f < n; ++f) {
                t0 = std::chrono::steady_clock::now();
                rgba_of_rgb(buf.data() + f * fb, sw * sh, rgba);
                gpu::copy(pic.p, rgba.data(), rgba.size());
                image::lanczos_resize_device(pic.p, sw, sh, sized.p, w, h);
                uint8_t* fd = feed.p;
                const uint8_t* zd = sized.p;
                done(queue().parallel_for(sycl::range<2>((size_t) fh, (size_t) fw), [=](sycl::id<2> id) {
                    const int64_t y = (int64_t) id[0], x = (int64_t) id[1];
                    const int64_t yy = sycl::min(y, h - 1), xx = sycl::min(x, w - 1);
                    for (int c = 0; c < 4; ++c) fd[(y * fw + x) * 4 + c] = zd[(yy * w + xx) * 4 + c];
                }), "upscale: padding");
                rep.resize_s += since(t0);
                waiting.emplace_back(buf.begin() + f * fb, buf.begin() + (f + 1) * fb);
                t0 = std::chrono::steady_clock::now();
                m.video_push(feed.p, 1);
                rep.model_s += since(t0);
                drain();
            }
        }
        const auto t0 = std::chrono::steady_clock::now();
        m.video_end();
        rep.model_s += since(t0);
        drain();
        m.release(false);
        } catch (...) {
            m.release(false);
            throw;
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    wr.finish();
    rep.write_s += since(t0);
    rep.path = wr.path();
    rep.mime = wr.mime();
    return rep;
}

}  // namespace strata::upscale
