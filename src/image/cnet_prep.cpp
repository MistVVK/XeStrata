// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/cnet_prep.cpp - see include/strata/image/cnet_prep.hpp.

#include "strata/image/cnet_prep.hpp"

#include "strata/image/cnet_models.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace strata::image {

// the kernels' names: a namespace of its own (an anonymous one mangles the same in every file)
namespace cnet_prep_k {

inline sycl::queue& queue() { return core::Runtime::get().stream(nullptr); }

// a run's buffers, kept from run to run: the n-th asked of a run takes the n-th slot (grown where it is short)
struct Pool {
    std::vector<std::pair<void*, size_t>> slots;
    size_t next = 0;
    template <class T> T* get(size_t n) {
        const size_t bytes = std::max<size_t>(n * sizeof(T), 16);
        if (next == slots.size()) slots.emplace_back(nullptr, 0);
        auto& s = slots[next++];
        if (s.second < bytes) {
            if (s.first != nullptr) {
                queue().wait();
                gpu::free(s.first);
            }
            s = {nullptr, 0};
            if (!gpu::alloc_device(&s.first, bytes))
                throw std::runtime_error(std::string("image engine (control): GPU memory: ") + gpu::last_error());
            s.second = bytes;
        }
        return static_cast<T*>(s.first);
    }
    size_t bytes() const {
        size_t n = 0;
        for (const auto& s : slots) n += s.second;
        return n;
    }
    void clear() {
        queue().wait();
        for (auto& s : slots)
            if (s.first != nullptr) gpu::free(s.first);
        slots.clear();
        next = 0;
    }
};

// scribble_xdog's centre and the surround's ratio to it
constexpr double kXdogSigma = 1.0, kXdogK = 4.0;

// a side to the nearest multiple of 64, 64 at least
inline int64_t round64(double x) { return std::max<int64_t>(64, std::llround(x / 64.0) * 64); }

// the luma of an RGB pixel (ITU-R BT.601)
inline float luma(const uint8_t* p) { return 0.299f * (float) p[0] + 0.587f * (float) p[1] + 0.114f * (float) p[2]; }

// ---- the picture at the processed size

// out [h][w][3] = the picture `in` [H][W][3] resampled to w x h, rounded to 8 bits: `area`, each output pixel the
// mean of the source area it covers (each source pixel weighted by the part covered); else Keys's cubic convolution
// (a = -0.5) at the pixel centres, the edges repeated.
//
// This one stays beside the shared Pillow resize (src/image/img_crop.hpp) instead of joining it: neither branch is one
// of its filters.  The cubic here keeps 4 taps whatever the scale, where Pillow's bicubic widens its support by the
// scale, and the two are 52% of a 1024 x 1024 picture's values apart at 384 x 384 (by up to 91 levels); Pillow's
// OpenCV cubic (cv_cubic) is the same shape with a = -0.75, still 21% apart (up to 10 levels); and the area branch,
// a box mean of the covered area, has no Pillow filter here at all.  The resized picture is what a ControlNet was
// trained to see, so folding either into the shared piece would move every control map's pixels.
void resample(const uint8_t* in, int64_t W, int64_t H, int64_t w, int64_t h, bool area, uint8_t* out) {
    const float sx = (float) W / (float) w, sy = (float) H / (float) h;
    queue().parallel_for(sycl::range<2>((size_t) h, (size_t) w), [=](sycl::id<2> id) {
        const auto y = static_cast<int64_t>(id[0]), x = static_cast<int64_t>(id[1]);
        float c[3] = {0.f, 0.f, 0.f};
        if (area) {
            const float x0 = (float) x * sx, x1 = x0 + sx, y0 = (float) y * sy, y1 = y0 + sy;
            const int64_t i0 = (int64_t) sycl::floor(x0), i1 = sycl::min((int64_t) sycl::ceil(x1), W);
            const int64_t j0 = (int64_t) sycl::floor(y0), j1 = sycl::min((int64_t) sycl::ceil(y1), H);
            float total = 0.f;
            for (int64_t j = j0; j < j1; ++j) {
                const float wy = sycl::fmin((float) (j + 1), y1) - sycl::fmax((float) j, y0);
                for (int64_t i = i0; i < i1; ++i) {
                    const float wt = wy * (sycl::fmin((float) (i + 1), x1) - sycl::fmax((float) i, x0));
                    const uint8_t* p = in + (j * W + i) * 3;
                    for (int k = 0; k < 3; ++k) c[k] += wt * (float) p[k];
                    total += wt;
                }
            }
            for (float& v : c) v /= total;
        } else {
            const float fx = ((float) x + 0.5f) * sx - 0.5f, fy = ((float) y + 0.5f) * sy - 0.5f;
            const int64_t ix = (int64_t) sycl::floor(fx), iy = (int64_t) sycl::floor(fy);
            const float tx = fx - (float) ix, ty = fy - (float) iy;
            const auto cub = [](float t, float* wv) {
                constexpr float A = -0.5f;
                wv[0] = ((A * (t + 1.f) - 5.f * A) * (t + 1.f) + 8.f * A) * (t + 1.f) - 4.f * A;
                wv[1] = ((A + 2.f) * t - (A + 3.f)) * t * t + 1.f;
                wv[2] = ((A + 2.f) * (1.f - t) - (A + 3.f)) * (1.f - t) * (1.f - t) + 1.f;
                wv[3] = 1.f - wv[0] - wv[1] - wv[2];
            };
            float wx[4], wy[4];
            cub(tx, wx);
            cub(ty, wy);
            for (int b = 0; b < 4; ++b) {
                const int64_t j = sycl::clamp(iy - 1 + b, (int64_t) 0, H - 1);
                for (int a = 0; a < 4; ++a) {
                    const int64_t i = sycl::clamp(ix - 1 + a, (int64_t) 0, W - 1);
                    const uint8_t* p = in + (j * W + i) * 3;
                    for (int k = 0; k < 3; ++k) c[k] += wy[b] * wx[a] * (float) p[k];
                }
            }
        }
        uint8_t* o = out + (y * w + x) * 3;
        for (int k = 0; k < 3; ++k) o[k] = (uint8_t) sycl::clamp(sycl::rint(c[k]), 0.f, 255.f);
    });
}

// ---- canny

// each pixel's gradient by Sobel's 3 x 3 differences (the edges repeated) of the picture's luma (BT.601): g [H][W] =
// (d/dx, d/dy, its Euclidean length)
void sobel(const uint8_t* in, int64_t W, int64_t H, sycl::float3* g) {
    queue().parallel_for(sycl::range<2>((size_t) H, (size_t) W), [=](sycl::id<2> id) {
        const auto y = static_cast<int64_t>(id[0]), x = static_cast<int64_t>(id[1]);
        const auto lum = [&](int64_t yy, int64_t xx) {
            const uint8_t* p = in + (sycl::clamp(yy, (int64_t) 0, H - 1) * W + sycl::clamp(xx, (int64_t) 0, W - 1)) * 3;
            return luma(p);
        };
        float gx = 0.f, gy = 0.f;
        for (int d = -1; d <= 1; ++d) {
            const float wgt = d == 0 ? 2.f : 1.f;
            gx += wgt * (lum(y + d, x + 1) - lum(y + d, x - 1));
            gy += wgt * (lum(y + 1, x + d) - lum(y - 1, x + d));
        }
        g[y * W + x] = sycl::float3(gx, gy, sycl::sqrt(gx * gx + gy * gy));
    });
}

// Canny's non-maximum suppression and two thresholds: a pixel stays where its gradient's length is no less than its
// two neighbours' across the edge (the gradient's direction taken to the nearest of the four through the 8
// neighbours; 0 past the picture); lab 2 an edge (above high), 1 a candidate (above low), 0 none
void suppress(const sycl::float3* g, int64_t W, int64_t H, float low, float high, uint8_t* lab) {
    queue().parallel_for(sycl::range<2>((size_t) H, (size_t) W), [=](sycl::id<2> id) {
        const auto y = static_cast<int64_t>(id[0]), x = static_cast<int64_t>(id[1]);
        const sycl::float3 v = g[y * W + x];
        const float m = v.z();
        const auto len = [&](int64_t yy, int64_t xx) {
            return yy < 0 || yy >= H || xx < 0 || xx >= W ? 0.f : g[yy * W + xx].z();
        };
        uint8_t l = 0;
        if (m > low) {
            // the direction's octant: 0 along x, 2 along y, 1 and 3 the diagonals (y down)
            float ang = sycl::atan2(v.y(), v.x());
            if (ang < 0.f) ang += 3.14159265f;
            const int dir = (int) sycl::floor(ang / 0.78539816f + 0.5f) % 4;
            const int64_t dx = dir == 0 ? 1 : dir == 2 ? 0 : dir == 1 ? 1 : -1, dy = dir == 0 ? 0 : 1;
            if (m >= len(y + dy, x + dx) && m >= len(y - dy, x - dx)) l = m > high ? 2 : 1;
        }
        lab[y * W + x] = l;
    });
}

// the hysteresis: candidates next to an edge (8 neighbours) become edges, until none changes.  A work-group takes a
// tile of side `t` into local memory and runs it to its own end; edges cross from tile to tile launch by launch
// (`changed` set when a tile changed)
void hysteresis(uint8_t* lab, int64_t W, int64_t H, int t, int* changed) {
    const int64_t gw = (W + t - 1) / t * t, gh = (H + t - 1) / t * t;
    const int side = t + 2;
    queue().submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 1> tile(sycl::range<1>((size_t) (side * side)), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) gh, (size_t) gw), sycl::range<2>((size_t) t, (size_t) t)),
                       [=](sycl::nd_item<2> it) {
            const int ly = (int) it.get_local_id(0), lx = (int) it.get_local_id(1);
            const int64_t y0 = (int64_t) it.get_group(0) * t - 1, x0 = (int64_t) it.get_group(1) * t - 1;
            for (int i = ly * t + lx; i < side * side; i += t * t) {
                const int64_t yy = y0 + i / side, xx = x0 + i % side;
                tile[(size_t) i] = yy >= 0 && yy < H && xx >= 0 && xx < W ? lab[yy * W + xx] : 0;
            }
            sycl::group_barrier(it.get_group());
            const int me = (ly + 1) * side + lx + 1;
            const int64_t y = y0 + 1 + ly, x = x0 + 1 + lx;
            const bool inside = y < H && x < W;
            bool any = false;
            for (;;) {
                bool grew = false;
                if (inside && tile[(size_t) me] == 1) {
                    for (int dy = -1; dy <= 1 && !grew; ++dy)
                        for (int dx = -1; dx <= 1; ++dx)
                            if (tile[(size_t) me + (size_t) (dy * side) + (size_t) dx] == 2) {
                                grew = true;
                                break;
                            }
                }
                sycl::group_barrier(it.get_group());
                if (grew) tile[(size_t) me] = 2;
                sycl::group_barrier(it.get_group());
                any = any || grew;
                if (!sycl::any_of_group(it.get_group(), grew)) break;
            }
            if (inside) lab[y * W + x] = tile[(size_t) me];
            if (sycl::any_of_group(it.get_group(), any) && ly == 0 && lx == 0)
                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device>(*changed).store(1);
        });
    });
}

// ---- the other maps without weights

// the grey picture: the luma (ITU-R BT.601), rounded
void gray(const uint8_t* in, int64_t n, uint8_t* out) {
    queue().parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        out[id[0]] = (uint8_t) sycl::clamp(sycl::rint(luma(in + static_cast<int64_t>(id[0]) * 3)), 0.f, 255.f);
    });
}

// a drawing's own dark lines white on black: white where the luma is below the middle grey
void scribble(const uint8_t* in, int64_t n, uint8_t* out) {
    queue().parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        out[id[0]] = luma(in + static_cast<int64_t>(id[0]) * 3) < 128.f ? 255 : 0;
    });
}

// a normalized Gaussian kernel of deviation s
std::vector<float> gauss_kernel(double s) {
    const int n = 2 * (int) std::ceil(3.0 * s) + 1;   // three deviations each side
    std::vector<float> k((size_t) n);
    double sum = 0;
    for (int i = 0; i < n; ++i) {
        const double x = i - (n - 1) / 2.0;
        k[(size_t) i] = (float) std::exp(-x * x / (2 * s * s));
        sum += k[(size_t) i];
    }
    for (float& v : k) v = (float) (v / sum);
    return k;
}

// one direction of a separable blur of `in` [H][W][3] (FP32 values) by k, the border mirrored about its edge pixel
template <class In>
void blur_pass(const In* in, int64_t W, int64_t H, const float* k, int n, bool rows, float* out) {
    queue().parallel_for(sycl::range<2>((size_t) H, (size_t) W), [=](sycl::id<2> id) {
        const auto y = static_cast<int64_t>(id[0]), x = static_cast<int64_t>(id[1]);
        const int64_t len = rows ? W : H, at = rows ? x : y, r = n / 2;
        float c[3] = {0.f, 0.f, 0.f};
        for (int j = 0; j < n; ++j) {
            int64_t p = at + j - r;
            if (len == 1) p = 0;
            else {
                while (p < 0 || p >= len) p = p < 0 ? -p : 2 * (len - 1) - p;
            }
            const In* s = rows ? in + (y * W + p) * 3 : in + (p * W + x) * 3;
            for (int q = 0; q < 3; ++q) c[q] += k[j] * (float) s[q];
        }
        float* o = out + (y * W + x) * 3;
        for (int q = 0; q < 3; ++q) o[q] = c[q];
    });
}

// scribble_xdog's map: the extended difference of Gaussians (Winnemoeller et al. 2012) of the luma, t * G(k sigma) -
// G(sigma) with t 0.98, thresholded to white where a line darker than its surroundings passes thr
void xdog(const float* g1, const float* g2, int64_t n, float thr, uint8_t* out) {
    queue().parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const auto lum = [&](const float* g) { return 0.299f * g[i * 3] + 0.587f * g[i * 3 + 1] + 0.114f * g[i * 3 + 2]; };
        out[i] = 0.98f * lum(g2) - lum(g1) > thr ? 255 : 0;
    });
}

// a single-channel map [h][w] (or RGB with `c3`) to RGBA8 on the host
QiRef to_rgba(const uint8_t* d, int64_t w, int64_t h, bool c3) {
    std::vector<uint8_t> all((size_t) (w * h * (c3 ? 3 : 1)));
    queue().wait();
    gpu::copy(all.data(), d, all.size());
    QiRef r;
    r.width = w;
    r.height = h;
    r.rgba.resize((size_t) (w * h * 4));
    for (int64_t y = 0; y < h; ++y)
        for (int64_t x = 0; x < w; ++x) {
            uint8_t* o = &r.rgba[(size_t) ((y * w + x) * 4)];
            for (int k = 0; k < 3; ++k) o[k] = all[(size_t) ((y * w + x) * (c3 ? 3 : 1) + (c3 ? k : 0))];
            o[3] = 255;
        }
    return r;
}

// the picture's alpha over white, 8 bits (truncated)
std::vector<uint8_t> over_white(const QiRef& img) {
    const int64_t n = img.width * img.height;
    std::vector<uint8_t> rgb((size_t) (n * 3));
    for (int64_t i = 0; i < n; ++i) {
        const float a = (float) img.rgba[(size_t) (i * 4 + 3)] / 255.f;
        for (int k = 0; k < 3; ++k)
            rgb[(size_t) (i * 3 + k)] =
                (uint8_t) std::clamp((float) img.rgba[(size_t) (i * 4 + k)] * a + 255.f * (1.f - a), 0.f, 255.f);
    }
    return rgb;
}


}  // namespace cnet_prep_k

using namespace cnet_prep_k;

std::vector<uint8_t> control_resize(const QiRef& img, int64_t w, int64_t h) {
    if (img.width <= 0 || img.height <= 0 || img.rgba.size() != (size_t) (img.width * img.height * 4))
        throw std::invalid_argument("control: an empty control image");
    const std::vector<uint8_t> rgb = over_white(img);
    uint8_t *src = nullptr, *dst = nullptr;
    if (!gpu::alloc_device(&src, rgb.size()) || !gpu::alloc_device(&dst, (size_t) (w * h * 3))) {
        gpu::free(src);
        throw std::runtime_error(std::string("image engine (control): GPU memory: ") + gpu::last_error());
    }
    gpu::copy(src, rgb.data(), rgb.size());
    resample(src, img.width, img.height, w, h, false, dst);
    QiRef r = to_rgba(dst, w, h, true);
    gpu::free(src);
    gpu::free(dst);
    return std::move(r.rgba);
}


struct ControlPrep::State {
    Pool pool;
    int tile = 0;   // the hysteresis' tile side (from the device's work-group limit)
};

ControlPrep::ControlPrep() : st_(std::make_unique<State>()) {}
ControlPrep::~ControlPrep() { release(); }

const std::vector<ControlKind>& ControlPrep::kinds() {
    static const std::vector<ControlKind> all = {
        {"canny", ""},          {"gray", ""},      {"scribble", ""},       {"scribble_xdog", ""},
        {"scribble_hed", "hed"}, {"hed", "hed"},    {"lineart", "lineart"}, {"lineart_coarse", "lineart_coarse"},
        {"mlsd", "mlsd"},       {"depth", "depth"}, {"pose", "pose"},       {"normal", "normal"},
        {"segmentation", "segmentation"}};
    return all;
}

const ControlKind* ControlPrep::find(const std::string& kind) {
    for (const ControlKind& k : kinds())
        if (k.name == kind) return &k;
    return nullptr;
}

uint64_t ControlPrep::work_bytes(int64_t width, int64_t height, const ControlParams& p) {
    const double k = (double) p.resolution / (double) std::min(width, height);
    const auto n = (uint64_t) (round64((double) width * k) * round64((double) height * k));
    // the picture, the resized one, the gradients or the two blurs, the map
    return (uint64_t) (width * height * 3) + n * (3 + 24 + 1) + 256;
}

void ControlPrep::release() { st_->pool.clear(); }

QiRef ControlPrep::run(const std::string& kind, const QiRef& img, const ControlParams& p, const ControlNets& nets) {
    if (find(kind) == nullptr) throw std::invalid_argument("control: no preprocessor " + kind);
    if (img.width <= 0 || img.height <= 0 || img.rgba.size() != (size_t) (img.width * img.height * 4))
        throw std::invalid_argument("control: an empty picture");
    if (p.resolution < 64 || p.resolution > 8192) throw std::invalid_argument("control: resolution 64 .. 8192");
    State& s = *st_;
    s.pool.next = 0;
    if (s.tile == 0) {
        const size_t wg = queue().get_device().get_info<sycl::info::device::max_work_group_size>();
        s.tile = 16;
        while (s.tile > 1 && (size_t) s.tile * (size_t) s.tile > wg) s.tile /= 2;
    }
    const int64_t W = img.width, H = img.height;
    const std::vector<uint8_t> rgb = over_white(img);
    // the processed size: the picture scaled for its shorter side to be `resolution`, each side then taken to the
    // nearest multiple of 64 (the networks' strides divide it), shrunk by area or grown by cubic convolution
    const double k = (double) p.resolution / (double) std::min(W, H);
    const int64_t pw = round64((double) W * k), ph = round64((double) H * k), n = pw * ph;
    const bool shrink = pw <= W && ph <= H;
    const auto need = [&](auto* net, const char* what) {
        if (net == nullptr) throw std::invalid_argument(std::string("control: ") + kind + " needs " + what + "'s weights");
        return net;
    };
    if (kind == "depth") {   // the network at the picture's own size, its map then resized as the others' pictures
        const std::vector<float> dep = need(nets.depth, "Depth Anything V2")->run(rgb, W, H);
        // the relative depth stretched over 0 .. 255 (near white)
        const auto [lo, hi] = std::minmax_element(dep.begin(), dep.end());
        const float a = *lo, span = *hi - *lo;
        std::vector<uint8_t> m((size_t) (W * H * 3));
        for (size_t i = 0; i < dep.size(); ++i) {
            const auto v = (uint8_t) (span > 0.f ? (dep[i] - a) / span * 255.f : 0.f);
            m[i * 3] = m[i * 3 + 1] = m[i * 3 + 2] = v;
        }
        auto* src = s.pool.get<uint8_t>(m.size());
        gpu::copy(src, m.data(), m.size());
        auto* pic = s.pool.get<uint8_t>((size_t) (n * 3));
        resample(src, W, H, pw, ph, shrink, pic);
        return to_rgba(pic, pw, ph, true);
    }
    auto* src = s.pool.get<uint8_t>(rgb.size());
    gpu::copy(src, rgb.data(), rgb.size());
    auto* pic = s.pool.get<uint8_t>((size_t) (n * 3));
    resample(src, W, H, pw, ph, shrink, pic);
    const auto host_pic = [&] {
        std::vector<uint8_t> v((size_t) (n * 3));
        queue().wait();
        gpu::copy(v.data(), pic, v.size());
        return v;
    };
    auto* map = s.pool.get<uint8_t>((size_t) n);
    const auto from_host = [&](const std::vector<uint8_t>& m, bool c3) {
        auto* d = s.pool.get<uint8_t>(m.size());
        gpu::copy(d, m.data(), m.size());
        return to_rgba(d, pw, ph, c3);
    };
    if (kind == "normal")
        return from_host(need(nets.normal, "BAE normal")->run(host_pic(), pw, ph), true);
    if (kind == "segmentation")
        return from_host(need(nets.segmentation, "UniFormer segmentation")->run(host_pic(), pw, ph), true);
    if (kind == "hed" || kind == "scribble_hed")
        return from_host(need(nets.hed, "HED")->run(host_pic(), pw, ph, p.safe, kind != "hed"), false);
    if (kind == "lineart" || kind == "lineart_coarse")
        return from_host(need(nets.lineart, "line art")->run(host_pic(), pw, ph), false);
    if (kind == "mlsd")
        return from_host(need(nets.mlsd, "M-LSD")->run(host_pic(), pw, ph, p.score, p.distance), false);
    if (kind == "pose") {
        const std::vector<uint8_t> v = host_pic();
        const std::vector<ControlBox> people = need(nets.yolox, "YOLOX")->people(v, pw, ph);
        const std::vector<WholeBody> bodies = people.empty() ? std::vector<WholeBody>()
                                                             : need(nets.dwpose, "DWPose")->run(v, pw, ph, people);
        return from_host(draw_pose(bodies, pw, ph, p.body, p.hands, p.face), true);
    }
    if (kind == "gray") {
        gray(pic, n, map);
    } else if (kind == "scribble") {
        scribble(pic, n, map);
    } else if (kind == "scribble_xdog") {
        if (p.threshold < 1 || p.threshold > 255) throw std::invalid_argument("control: threshold 1 .. 255");
        auto* tmp = s.pool.get<float>((size_t) (n * 3));
        auto* g1 = s.pool.get<float>((size_t) (n * 3));
        auto* g2 = s.pool.get<float>((size_t) (n * 3));
        for (const auto& [sigma, out] : {std::pair<double, float*>{kXdogSigma, g1}, {kXdogSigma * kXdogK, g2}}) {
            const std::vector<float> kh = gauss_kernel(sigma);
            auto* kd = s.pool.get<float>(kh.size());
            gpu::copy(kd, kh.data(), kh.size() * 4);
            blur_pass(pic, pw, ph, kd, (int) kh.size(), true, tmp);
            blur_pass(tmp, pw, ph, kd, (int) kh.size(), false, out);
        }
        xdog(g1, g2, n, p.threshold, map);
    } else {   // canny
        if (p.low < 0 || p.high < 0 || p.low > 1020 || p.high > 1020)
            throw std::invalid_argument("control: canny's thresholds 0 .. 1020");
        const float lo = std::min(p.low, p.high), hi = std::max(p.low, p.high);
        auto* g = s.pool.get<sycl::float3>((size_t) n);
        sobel(pic, pw, ph, g);
        suppress(g, pw, ph, lo, hi, map);
        int* changed = s.pool.get<int>(1);
        for (int it = 0;; ++it) {
            int flag = 0;
            gpu::copy(changed, &flag, sizeof flag);
            hysteresis(map, pw, ph, s.tile, changed);
            queue().wait();
            gpu::copy(&flag, changed, sizeof flag);
            if (flag == 0 || it > (int) (pw + ph)) break;
        }
        queue().parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
            map[id[0]] = map[id[0]] == 2 ? 255 : 0;
        });
    }
    return to_rgba(map, pw, ph, false);
}

}  // namespace strata::image
