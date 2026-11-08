// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/yolo_detect.cpp - see include/strata/image/yolo_detect.hpp.

#include "strata/image/yolo_detect.hpp"

#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/kernels/f16_bits.hpp"

#if STRATA_ONEDNN
#include "qi_conv.hpp"
#endif

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace strata::image {

namespace {

using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;

constexpr int64_t kColBytes = 64ll << 20;   // the unfolded neighbourhoods a product takes at most
constexpr int kInC = 8;                     // the picture's channels on the GPU: RGB and zeros (8-wide loads)

sycl::queue& queue() { return core::Runtime::get().stream(nullptr); }

int64_t up8(int64_t n) { return (n + 7) / 8 * 8; }

// a convolution with its batch norm folded in: w [outp][k][k][in] (FP16, rows past `out` zero), b [outp]
struct ConvW {
    int64_t in = 0, out = 0, outp = 0, k = 1, s = 1;
    uint16_t* w = nullptr;
    float* b = nullptr;
    bool act = true;   // SiLU after it (the head's last 1 x 1 convolutions have none)
};

struct Layer {
    std::string mod;
    std::vector<int> from;
    int stride = 1;
    bool shortcut = false;
    ConvW conv, cv1, cv2;                         // Conv: conv; C2f and SPPF: cv1, cv2
    std::vector<std::pair<ConvW, ConvW>> m;       // C2f's bottlenecks
};

// a buffer list reused call after call: the n-th request of a detection takes the n-th slot
struct Pool {
    std::vector<std::pair<void*, size_t>> slots;
    size_t next = 0;
    template <class T> T* get(size_t n) {
        const size_t bytes = std::max<size_t>(n * sizeof(T), 16);
        if (next == slots.size()) slots.emplace_back(nullptr, 0);
        auto& s = slots[next++];
        if (s.second < bytes) {
            if (s.first != nullptr) gpu::free(s.first);
            s.first = nullptr;
            s.second = 0;
            if (!gpu::alloc_device(&s.first, bytes))
                throw std::runtime_error(std::string("image engine (detector): GPU memory: ") + gpu::last_error());
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
        for (auto& s : slots)
            if (s.first != nullptr) gpu::free(s.first);
        slots.clear();
        next = 0;
    }
};

// a feature map on the GPU: [h][w][c] FP16
struct Map {
    uint16_t* p = nullptr;
    int64_t h = 0, w = 0, c = 0;
};

// ---- kernels

// the picture (RGBA8, W x H) letterboxed: resized to nw x nh by OpenCV's bilinear (pixel centres, the edges
// clamped, rounded to 8 bits), placed at (left, top) of an iw x ih field of grey 114, as RGB / 255 and zeros in kInC
void letterbox(const uint8_t* rgba, int64_t W, int64_t H, int64_t nw, int64_t nh, int64_t left, int64_t top,
               int64_t iw, int64_t ih, uint16_t* out) {
    const float sx = (float) W / (float) nw, sy = (float) H / (float) nh;
    queue().parallel_for(sycl::range<2>((size_t) ih, (size_t) iw), [=](sycl::id<2> id) {
        const auto y = static_cast<int64_t>(id[0]), x = static_cast<int64_t>(id[1]);
        uint16_t* o = out + (y * iw + x) * kInC;
        const int64_t u = x - left, v = y - top;
        float c[3] = {114.f, 114.f, 114.f};
        if (u >= 0 && u < nw && v >= 0 && v < nh) {
            float fx = ((float) u + 0.5f) * sx - 0.5f, fy = ((float) v + 0.5f) * sy - 0.5f;
            int64_t x0 = (int64_t) sycl::floor(fx), y0 = (int64_t) sycl::floor(fy);
            fx -= (float) x0;
            fy -= (float) y0;
            if (x0 < 0) { x0 = 0; fx = 0.f; }
            if (y0 < 0) { y0 = 0; fy = 0.f; }
            if (x0 >= W - 1) { x0 = W - 1; fx = 0.f; }
            if (y0 >= H - 1) { y0 = H - 1; fy = 0.f; }
            const int64_t x1 = sycl::min(x0 + 1, W - 1), y1 = sycl::min(y0 + 1, H - 1);
            for (int k = 0; k < 3; ++k) {
                const float a = rgba[(y0 * W + x0) * 4 + k], b = rgba[(y0 * W + x1) * 4 + k];
                const float d = rgba[(y1 * W + x0) * 4 + k], e = rgba[(y1 * W + x1) * 4 + k];
                const float top_ = a + (b - a) * fx, bot = d + (e - d) * fx;
                c[k] = sycl::round(top_ + (bot - top_) * fy);
            }
        }
        for (int k = 0; k < kInC; ++k) o[k] = f16_from_f32(k < 3 ? c[k] / 255.f : 0.f);
    });
}

// a convolution's output y [HW][ldy] (FP32) finished (qi_conv_store16, on the queue without a wait): + bias (null:
// in y already), SiLU, + res [HW][O] (null: none), to FP16 at dst [HW][ld] from channel `off`, and its channels from
// `from2` on also to dst2 [HW][O - from2] (null: none)
void store(const float* y, int64_t HW, int64_t O, int64_t ldy, const float* bias, bool silu, const uint16_t* res,
           uint16_t* dst, int64_t ld, int64_t off, uint16_t* dst2, int64_t from2) {
    qi_conv_store16(y, HW, O, ldy, bias, silu ? QiAct::silu : QiAct::none, 0.f, res, dst, ld, off,
                    &core::Runtime::get().compute(), dst2, from2);
}

// y [HW][O] += b [O]
void add_bias(float* y, int64_t HW, int64_t O, const float* b) { qi_bias_add(y, b, HW, O, nullptr); }

// the 5 x 5 max pool of stride 1 (padded by -inf) of x [H][W][C] to dst [HW][ld] from channel off and to dst2
// [HW][C]
void maxpool5(const uint16_t* x, int64_t H, int64_t W, int64_t C, uint16_t* dst, int64_t ld, int64_t off,
              uint16_t* dst2) {
    queue().parallel_for(sycl::range<1>((size_t) (H * W * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t q = i / C, c = i - q * C, y = q / W, xx = q % W;
        float m = -INFINITY;
        for (int64_t dy = -2; dy <= 2; ++dy)
            for (int64_t dx = -2; dx <= 2; ++dx) {
                const int64_t yy = y + dy, x2 = xx + dx;
                if (yy >= 0 && yy < H && x2 >= 0 && x2 < W) m = sycl::fmax(m, f32_from_f16(x[(yy * W + x2) * C + c]));
            }
        const uint16_t h = f16_from_f32(m);
        dst[q * ld + off + c] = h;
        if (dst2 != nullptr) dst2[q * C + c] = h;
    });
}

// dst [H][W][Ca + Cb] = (a, nearest 2x upsampled from H / 2 x W / 2 when `up`) and b
void concat2(const uint16_t* a, int64_t Ca, bool up, const uint16_t* b, int64_t Cb, int64_t H, int64_t W,
             uint16_t* dst) {
    const int64_t C = Ca + Cb;
    queue().parallel_for(sycl::range<1>((size_t) (H * W * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t q = i / C, c = i - q * C;
        if (c < Ca) {
            const int64_t src = up ? ((q / W) / 2) * (W / 2) + (q % W) / 2 : q;
            dst[i] = a[src * Ca + c];
        } else {
            dst[i] = b[q * Cb + c - Ca];
        }
    });
}

// a 2 x 2 transposed convolution of stride 2's product y [H][W][4][O] (the kernel's place (a, b) = a * 2 + b) laid
// out at 2H x 2W, + bias, to FP16 dst [2H][2W][O]
void unshuffle2(const float* y, int64_t H, int64_t W, int64_t O, const float* bias, uint16_t* dst) {
    queue().parallel_for(sycl::range<1>((size_t) (4 * H * W * O)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t q = i / O, o = i - q * O, Y = q / (2 * W), X = q % (2 * W);
        const int64_t src = ((Y / 2) * W + X / 2) * 4 + (Y % 2) * 2 + (X % 2);
        dst[i] = f16_from_f32(y[src * O + o] + bias[o]);
    });
}

// one level's head decoded into out[a0 + q][6 + nm]: the box in the letterboxed input (x0, y0, x1, y1: the anchor
// point (x + 0.5, y + 0.5) less and plus the four distances, each the expectation of its 16 bins' softmax by the
// distribution focal loss weights, times the stride), the best class's sigmoid and the class, then the mask
// coefficients
void decode(const float* box, const float* cls, int64_t ldc, const float* coef, int64_t H, int64_t W, float stride,
            int nc, int nm, const float* dfl, int reg_max, int64_t a0, float* out) {
    const int64_t row = 6 + nm;
    queue().parallel_for(sycl::range<1>((size_t) (H * W)), [=](sycl::id<1> id) {
        const auto q = static_cast<int64_t>(id[0]);
        const int64_t qx = q % W, qy = q / W;
        const float ax = (float) qx + 0.5f, ay = (float) qy + 0.5f;
        float d[4];
        for (int j = 0; j < 4; ++j) {
            const float* l = box + (q * 4 + j) * (int64_t) reg_max;
            float m = l[0];
            for (int k = 1; k < reg_max; ++k) m = sycl::fmax(m, l[k]);
            float s = 0.f, e = 0.f;
            for (int k = 0; k < reg_max; ++k) {
                const float p = sycl::exp(l[k] - m);
                s += p;
                e += p * dfl[k];
            }
            d[j] = e / s;
        }
        float* o = out + (a0 + q) * row;
        o[0] = (ax - d[0]) * stride;
        o[1] = (ay - d[1]) * stride;
        o[2] = (ax + d[2]) * stride;
        o[3] = (ay + d[3]) * stride;
        int best = 0;
        float bs = cls[q * ldc];
        for (int c = 1; c < nc; ++c)
            if (cls[q * ldc + c] > bs) {
                bs = cls[q * ldc + c];
                best = c;
            }
        o[4] = 1.f / (1.f + sycl::exp(-bs));
        o[5] = (float) best;
        for (int k = 0; k < nm; ++k) o[6 + k] = coef[q * nm + k];
    });
}

// the masks at the picture's size: detection n's prototype product logit [Hp * Wp][ldn] (column n) through a
// sigmoid, sampled bilinearly (pixel centres, the edges clamped) at the place each picture pixel takes in the
// letterboxed input, 1 where it is above 0.5 inside the detection's box (boxes [n][4], the picture's pixels)
void masks_kernel(const float* logit, int64_t Hp, int64_t Wp, int64_t ldn, int64_t N, const float* boxes, float gain,
                  float left, float top, float in_w, float in_h, int64_t W, int64_t H, uint8_t* out) {
    const float px = (float) Wp / in_w, py = (float) Hp / in_h;
    queue().parallel_for(sycl::range<1>((size_t) (N * H * W)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t n = i / (H * W), q = i - n * H * W, y = q / W, x = q % W;
        const float* b = boxes + n * 4;
        const float cx = (float) x + 0.5f, cy = (float) y + 0.5f;
        if (cx < b[0] || cx >= b[2] || cy < b[1] || cy >= b[3]) {
            out[i] = 0;
            return;
        }
        const float u = (cx * gain + left) * px - 0.5f, v = (cy * gain + top) * py - 0.5f;
        float fu = sycl::clamp(u, 0.f, (float) (Wp - 1)), fv = sycl::clamp(v, 0.f, (float) (Hp - 1));
        const int64_t u0 = (int64_t) fu, v0 = (int64_t) fv;
        const int64_t u1 = sycl::min(u0 + 1, Wp - 1), v1 = sycl::min(v0 + 1, Hp - 1);
        fu -= (float) u0;
        fv -= (float) v0;
        const auto sg = [&](int64_t yy, int64_t xx) { return 1.f / (1.f + sycl::exp(-logit[(yy * Wp + xx) * ldn + n])); };
        const float t = sg(v0, u0) + (sg(v0, u1) - sg(v0, u0)) * fu, bt = sg(v1, u0) + (sg(v1, u1) - sg(v1, u0)) * fu;
        out[i] = t + (bt - t) * fv > 0.5f ? 1 : 0;
    });
}

std::string join(const std::string& a, const std::string& b) { return a + "." + b; }

}  // namespace

struct YoloDetector::Net {
    std::vector<Layer> layers;
    std::vector<int> head_from;
    struct Level {
        ConvW box[3], cls[3], coef[3];
    };
    std::vector<Level> lv;
    ConvW p1, p2, p3;            // the prototypes: before and after the transposed convolution, the last 1 x 1
    uint16_t* up_w = nullptr;    // the transposed convolution as a product: [4 * O][C] (place (a, b) = a * 2 + b)
    float* up_b = nullptr;
    int64_t up_c = 0;
    float* dfl = nullptr;        // the bins' weights [reg_max]
    int reg_max = 16, nm = 0;
    std::vector<void*> owned;
    Pool pool;
    ~Net() {
        for (void* p : owned) gpu::free(p);
        pool.clear();
    }
    template <class T> T* upload(const std::vector<T>& v) {
        T* p = nullptr;
        if (!gpu::alloc_device(&p, v.size() * sizeof(T)))
            throw std::runtime_error(std::string("image engine (detector): GPU memory: ") + gpu::last_error());
        owned.push_back(p);
        gpu::copy(p, v.data(), v.size() * sizeof(T));
        return p;
    }
};

YoloDetector::YoloDetector() = default;
YoloDetector::~YoloDetector() = default;

namespace {

// model.<i>.<...>'s convolution and batch norm (`bn`: a Conv module; else a plain Conv2d with its bias), folded
ConvW load_conv(SafeTensors& st, YoloDetector::Net& net, const std::string& p, float eps, bool bn, int64_t in_pad,
                uint64_t& bytes) {
    const std::string wn = bn ? join(p, "conv.weight") : join(p, "weight");
    const SafeTensor* t = st.find(wn);
    if (t == nullptr || t->shape.size() != 4 || t->shape[2] != t->shape[3])
        throw std::runtime_error("detector: no convolution " + wn);
    ConvW c;
    c.out = t->shape[0];
    c.in = t->shape[1];
    c.k = t->shape[2];
    c.act = bn;
    const int64_t inp = std::max(in_pad, c.in);
    if (inp % 8 != 0) throw std::runtime_error("detector: " + wn + ": " + std::to_string(c.in) + " input channels");
    c.outp = up8(c.out);
    const std::vector<float> w = st.f32(wn);
    std::vector<float> scale((size_t) c.out, 1.f), shift((size_t) c.out, 0.f);
    if (bn) {
        const auto g = st.f32(join(p, "bn.weight")), b = st.f32(join(p, "bn.bias"));
        const auto m = st.f32(join(p, "bn.running_mean")), v = st.f32(join(p, "bn.running_var"));
        if ((int64_t) g.size() != c.out || b.size() != g.size() || m.size() != g.size() || v.size() != g.size())
            throw std::runtime_error("detector: " + p + ": batch norm of another size");
        for (int64_t o = 0; o < c.out; ++o) {
            scale[o] = g[o] / std::sqrt(v[o] + eps);
            shift[o] = b[o] - m[o] * scale[o];
        }
    } else {
        const auto b = st.f32(join(p, "bias"));
        if ((int64_t) b.size() != c.out) throw std::runtime_error("detector: " + p + ": no bias");
        shift = b;
    }
    std::vector<uint16_t> w16((size_t) (c.outp * c.k * c.k * inp), 0);
    for (int64_t o = 0; o < c.out; ++o)
        for (int64_t i = 0; i < c.in; ++i)
            for (int64_t ky = 0; ky < c.k; ++ky)
                for (int64_t kx = 0; kx < c.k; ++kx)
                    w16[(size_t) (((o * c.k + ky) * c.k + kx) * inp + i)] =
                        f16_from_f32(w[(size_t) (((o * c.in + i) * c.k + ky) * c.k + kx)] * scale[o]);
    shift.resize((size_t) c.outp, 0.f);
    c.in = inp;
    c.w = net.upload(w16);
    c.b = net.upload(shift);
    bytes += w16.size() * 2 + shift.size() * 4;
    return c;
}

}  // namespace

bool YoloDetector::open(const std::string& path, std::string& err) {
    try {
        net_ = std::make_unique<Net>();
        Net& net = *net_;
        weight_bytes_ = 0;
        SafeTensors st(path);
        const std::string graph = st.meta("graph");
        if (graph.empty()) throw std::runtime_error(path + ": no layer list (convert it with tools/image/yolo_pt.py)");
        const std::string eps_s = st.meta("bn_eps");
        const float eps = eps_s.empty() ? 1e-3f : std::stof(eps_s);
        names_.clear();
        {   // {"0": "face", ...}: the names in index order
            const std::string n = st.meta("names");
            for (size_t i = 0; (i = n.find(':', i)) != std::string::npos; ++i) {
                const size_t a = n.find('"', i), b = a == std::string::npos ? a : n.find('"', a + 1);
                if (b == std::string::npos) break;
                names_.push_back(n.substr(a + 1, b - a - 1));
                i = b;
            }
        }
        std::stringstream gs(graph);
        for (std::string entry; std::getline(gs, entry, ';');) {
            std::istringstream es(entry);
            Layer L;
            int index = -1, shortcut = 0;
            std::string from;
            es >> index >> L.mod >> from >> L.stride >> shortcut;
            if (!es || index != (int) net.layers.size()) throw std::runtime_error(std::string(path).append(": layer list entry: ").append(entry));
            L.shortcut = shortcut != 0;
            std::stringstream fs(from);
            for (std::string v; std::getline(fs, v, ',');) {
                const int f = std::stoi(v);
                L.from.push_back(f < 0 ? index + f : f);
            }
            const std::string p = "model." + std::to_string(index);
            const int64_t in0 = index == 0 ? kInC : 0;
            if (L.mod == "Conv") {
                L.conv = load_conv(st, net, p, eps, true, in0, weight_bytes_);
                L.conv.s = L.stride;
            } else if (L.mod == "C2f" || L.mod == "SPPF") {
                L.cv1 = load_conv(st, net, join(p, "cv1"), eps, true, in0, weight_bytes_);
                L.cv2 = load_conv(st, net, join(p, "cv2"), eps, true, 0, weight_bytes_);
                for (int j = 0; L.mod == "C2f" && st.find(p + ".m." + std::to_string(j) + ".cv1.conv.weight"); ++j) {
                    const std::string b = p + ".m." + std::to_string(j);
                    L.m.emplace_back(load_conv(st, net, join(b, "cv1"), eps, true, 0, weight_bytes_),
                                     load_conv(st, net, join(b, "cv2"), eps, true, 0, weight_bytes_));
                }
                if (L.mod == "C2f" && (L.m.empty() || L.cv1.out % 2 != 0 ||
                                       L.cv2.in != L.cv1.out / 2 * (2 + (int64_t) L.m.size())))
                    throw std::runtime_error((path + ": ").append(p).append(": a C2f block of another shape"));
            } else if (L.mod == "Detect" || L.mod == "Segment") {
                if (index + 1 != (int) std::count(graph.begin(), graph.end(), ';') + 1)
                    throw std::runtime_error(path + ": the head is not the last layer");
                net.head_from = L.from;
                seg_ = L.mod == "Segment";
                for (size_t l = 0; l < L.from.size(); ++l) {
                    Net::Level lv;
                    const std::string ls = std::to_string(l);
                    for (int j = 0; j < 3; ++j) {
                        const auto name = [&](const char* branch) {   // model.<i>.<branch>.<level>.<j>
                            std::string n = p;
                            return n.append(branch).append(ls).append(".").append(std::to_string(j));
                        };
                        lv.box[j] = load_conv(st, net, name(".cv2."), eps, j < 2, 0, weight_bytes_);
                        lv.cls[j] = load_conv(st, net, name(".cv3."), eps, j < 2, 0, weight_bytes_);
                        if (seg_) lv.coef[j] = load_conv(st, net, name(".cv4."), eps, j < 2, 0, weight_bytes_);
                    }
                    net.lv.push_back(lv);
                }
                const auto dfl = st.f32(p + ".dfl.conv.weight");
                net.reg_max = (int) dfl.size();
                if (net.reg_max < 1 || net.lv[0].box[2].out != 4 * (int64_t) net.reg_max)
                    throw std::runtime_error(path + ": a box head of another shape");
                net.dfl = net.upload(dfl);
                nc_ = (int) net.lv[0].cls[2].out;
                if (seg_) {
                    net.nm = (int) net.lv[0].coef[2].out;
                    net.p1 = load_conv(st, net, p + ".proto.cv1", eps, true, 0, weight_bytes_);
                    net.p2 = load_conv(st, net, p + ".proto.cv2", eps, true, 0, weight_bytes_);
                    net.p3 = load_conv(st, net, p + ".proto.cv3", eps, true, 0, weight_bytes_);
                    const SafeTensor* t = st.find(p + ".proto.upsample.weight");
                    if (t == nullptr || t->shape.size() != 4 || t->shape[2] != 2 || t->shape[3] != 2 ||
                        t->shape[0] != net.p1.out || t->shape[1] != net.p2.in || net.p3.out != net.nm)
                        throw std::runtime_error(path + ": a prototype branch of another shape");
                    const int64_t C = t->shape[0], O = t->shape[1];
                    const auto w = st.f32(p + ".proto.upsample.weight");
                    std::vector<uint16_t> w16((size_t) (4 * O * C));
                    for (int64_t c = 0; c < C; ++c)
                        for (int64_t o = 0; o < O; ++o)
                            for (int64_t ab = 0; ab < 4; ++ab)
                                w16[(size_t) ((ab * O + o) * C + c)] = f16_from_f32(w[(size_t) ((c * O + o) * 4 + ab)]);
                    net.up_w = net.upload(w16);
                    net.up_b = net.upload(st.f32(p + ".proto.upsample.bias"));
                    net.up_c = O;
                    weight_bytes_ += w16.size() * 2 + (size_t) O * 4;
                }
            } else if (L.mod != "Upsample" && L.mod != "Concat") {
                throw std::runtime_error(path + ": a module the detector does not take: " + L.mod);
            }
            net.layers.push_back(std::move(L));
        }
        if (net.lv.empty()) throw std::runtime_error(path + ": no detection head");
        if (!gemm_.init(nullptr, 16, err)) return false;
        path_ = path;
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        net_.reset();
        return false;
    }
}

uint64_t YoloDetector::work_bytes(int imgsz) const {
    if (!net_) return 0;
    // the feature maps: every layer's output kept (the neck reads back), at most the first convolution's (s / 2 of the
    // input's side) times the widest block, as FP16 and FP32, plus the unfolded neighbourhoods and the masks' product
    uint64_t widest = 0;
    for (const Layer& L : net_->layers) widest = std::max<uint64_t>(widest, (uint64_t) std::max(L.conv.outp, L.cv2.in));
    const uint64_t side = (uint64_t) imgsz / 2;
    return side * side * widest * (2 + 4) * 2 + (uint64_t) kColBytes + (seg_ ? (uint64_t) imgsz * imgsz * 64 : 0);
}

void YoloDetector::drop_scratch() {
    if (net_) net_->pool.clear();
}

std::vector<Detection> YoloDetector::detect(const QiRef& img, float conf, std::vector<std::vector<uint8_t>>* masks,
                                            float iou, int max_det, int imgsz) {
    if (!net_) throw std::runtime_error("detector: not open");
    Net& net = *net_;
    const int64_t W = img.width, H = img.height;
    if (W <= 0 || H <= 0 || (int64_t) img.rgba.size() != W * H * 4) throw std::invalid_argument("detector: a picture of another size");
    net.pool.next = 0;
    sycl::queue& q = queue();

    // the letterbox: the least padding to a multiple of 32, its smaller half before, grey 114 (YOLOv8's
    // letterbox)
    const double r = std::min((double) imgsz / (double) H, (double) imgsz / (double) W);
    const int64_t nw = (int64_t) std::lround((double) W * r), nh = (int64_t) std::lround((double) H * r);
    const int64_t pw = (imgsz - nw) % 32, ph = (imgsz - nh) % 32, iw = nw + pw, ih = nh + ph;
    const int64_t left = pw / 2, top = ph / 2;
    uint8_t* src = net.pool.get<uint8_t>((size_t) (W * H * 4));
    q.memcpy(src, img.rgba.data(), (size_t) (W * H * 4));
    Map x0{net.pool.get<uint16_t>((size_t) (iw * ih * kInC)), ih, iw, kInC};
    letterbox(src, W, H, nw, nh, left, top, iw, ih, x0.p);

    float* y32 = nullptr;   // the convolutions' FP32 output, the largest one's size
    size_t y32_cap = 0;
    uint16_t* col = nullptr;
    const auto y_buf = [&](size_t n) {
        if (n > y32_cap) {
            y32 = net.pool.get<float>(n);
            y32_cap = n;
        }
        return y32;
    };
    // y [OH][OW][c.outp] (FP32) = the convolution of x; whether the bias is in y already
    const auto conv = [&](const ConvW& c, const Map& x, int64_t& OH, int64_t& OW, float* yout = nullptr) -> bool {
        if (x.c != c.in) throw std::runtime_error("detector: a convolution of " + std::to_string(c.in) + " inputs on " +
                                                  std::to_string(x.c) + " channels");
        const int p = (int) c.k / 2;
        OH = (x.h + 2 * (int64_t) p - c.k) / c.s + 1;
        OW = (x.w + 2 * (int64_t) p - c.k) / c.s + 1;
        float* y = yout != nullptr ? yout : y_buf((size_t) (OH * OW * c.outp));
#if STRATA_ONEDNN
        if (c.k > 1 && conv::onednn(q, x.p, x.h, x.w, c.in, c.w, c.outp, (int) c.k, (int) c.s, p, p, p, p, c.b, false, y))
            return true;
#endif
        if (c.k == 1 && c.s == 1) {
            gemm_.f16(x.p, c.w, y, x.h * x.w, c.outp, c.in);
            return false;
        }
        const int64_t K = c.k * c.k * c.in, rows = std::max<int64_t>(1, kColBytes / 2 / K), HW = OH * OW;
        if (col == nullptr) col = net.pool.get<uint16_t>((size_t) (kColBytes / 2));
        for (int64_t p0 = 0; p0 < HW; p0 += rows) {
            const int64_t n = std::min(rows, HW - p0);
            qi_im2col16(x.p, 1, x.h, x.w, c.in, 1, (int) c.k, (int) c.s, p, OH, OW, p0, n, col, nullptr);
            gemm_.f16(col, c.w, y + p0 * c.outp, n, c.outp, K);
        }
        return false;
    };
    // a Conv module: the convolution, then SiLU into a new map (and from channel `from2` on into `dst2`)
    const auto conv_map = [&](const ConvW& c, const Map& x, uint16_t* dst = nullptr, int64_t ld = 0, int64_t off = 0,
                              uint16_t* dst2 = nullptr, int64_t from2 = 0, const uint16_t* res = nullptr) {
        int64_t OH = 0, OW = 0;
        const bool biased = conv(c, x, OH, OW);
        Map out{dst, OH, OW, c.out};
        if (out.p == nullptr) {
            out.p = net.pool.get<uint16_t>((size_t) (OH * OW * c.out));
            ld = c.out;
            off = 0;
        }
        store(y32, OH * OW, c.out, c.outp, biased ? nullptr : c.b, c.act, res, out.p, ld, off, dst2, from2);
        return out;
    };

    std::vector<Map> outs;
    outs.reserve(net.layers.size());
    Map x = x0;
    for (size_t i = 0; i < net.layers.size(); ++i) {
        const Layer& L = net.layers[i];
        const Map in = L.from.empty() || L.from[0] < 0 ? x : outs[(size_t) L.from[0]];
        if (L.mod == "Conv") {
            x = conv_map(L.conv, in);
        } else if (L.mod == "C2f") {
            const int64_t c = L.cv1.out / 2, n = (int64_t) L.m.size(), Ccat = (2 + n) * c;
            Map cat{net.pool.get<uint16_t>((size_t) (in.h * in.w * Ccat)), in.h, in.w, Ccat};
            Map last{net.pool.get<uint16_t>((size_t) (in.h * in.w * c)), in.h, in.w, c};
            conv_map(L.cv1, in, cat.p, Ccat, 0, last.p, c);
            for (int64_t j = 0; j < n; ++j) {
                const Map t = conv_map(L.m[(size_t) j].first, last);
                Map next{net.pool.get<uint16_t>((size_t) (in.h * in.w * c)), in.h, in.w, c};
                conv_map(L.m[(size_t) j].second, t, cat.p, Ccat, (2 + j) * c, next.p, 0, L.shortcut ? last.p : nullptr);
                last = next;
            }
            x = conv_map(L.cv2, cat);
        } else if (L.mod == "SPPF") {
            const int64_t c = L.cv1.out, Ccat = 4 * c;
            Map cat{net.pool.get<uint16_t>((size_t) (in.h * in.w * Ccat)), in.h, in.w, Ccat};
            Map a{net.pool.get<uint16_t>((size_t) (in.h * in.w * c)), in.h, in.w, c};
            conv_map(L.cv1, in, cat.p, Ccat, 0, a.p, 0);
            for (int j = 1; j < 4; ++j) {
                Map b{net.pool.get<uint16_t>((size_t) (in.h * in.w * c)), in.h, in.w, c};
                maxpool5(a.p, in.h, in.w, c, cat.p, Ccat, j * c, b.p);
                a = b;
            }
            x = conv_map(L.cv2, cat);
        } else if (L.mod == "Upsample") {   // the map as it is, at twice its size: the Concat after it reads it so
            x = Map{in.p, in.h * 2, in.w * 2, in.c};
        } else if (L.mod == "Concat") {
            if (L.from.size() != 2) throw std::runtime_error("detector: a Concat of " + std::to_string(L.from.size()));
            const Map& b = outs[(size_t) L.from[1]];
            const bool up = i > 0 && net.layers[i - 1].mod == "Upsample" && L.from[0] == (int) i - 1;
            if (in.h != b.h || in.w != b.w) throw std::runtime_error("detector: a Concat of maps of other sizes");
            Map cat{net.pool.get<uint16_t>((size_t) (b.h * b.w * (in.c + b.c))), b.h, b.w, in.c + b.c};
            concat2(in.p, in.c, up, b.p, b.c, b.h, b.w, cat.p);
            x = cat;
        } else {   // the head
            break;
        }
        outs.push_back(x);
    }

    // the head: each level's three branches, decoded into one list of candidates
    const int nm = seg_ ? net.nm : 0, row = 6 + nm;
    int64_t anchors = 0;
    for (int f : net.head_from) anchors += outs[(size_t) f].h * outs[(size_t) f].w;
    float* cand = net.pool.get<float>((size_t) (anchors * row));
    int64_t a0 = 0;
    for (size_t l = 0; l < net.lv.size(); ++l) {
        const Map& f = outs[(size_t) net.head_from[l]];
        const int64_t HW = f.h * f.w;
        const float stride = (float) ih / (float) f.h;
        float* br[3] = {nullptr, nullptr, nullptr};
        int64_t ldc = 0;
        const ConvW* chains[3] = {net.lv[l].box, net.lv[l].cls, seg_ ? net.lv[l].coef : nullptr};
        for (int k = 0; k < (seg_ ? 3 : 2); ++k) {
            const Map t = conv_map(chains[k][1], conv_map(chains[k][0], f));
            const ConvW& last = chains[k][2];
            br[k] = net.pool.get<float>((size_t) (HW * last.outp));
            int64_t OH = 0, OW = 0;
            if (!conv(last, t, OH, OW, br[k])) add_bias(br[k], HW, last.outp, last.b);
            if (k == 1) ldc = last.outp;
        }
        decode(br[0], br[1], ldc, br[2], f.h, f.w, stride, nc_, nm, net.dfl, net.reg_max, a0, cand);
        a0 += HW;
    }
    std::vector<float> host((size_t) (anchors * row));
    q.memcpy(host.data(), cand, host.size() * sizeof(float)).wait();

    // the candidates above the threshold, highest first, through non-maximum suppression (every class together)
    std::vector<int64_t> idx;
    for (int64_t a = 0; a < anchors; ++a)
        if (host[(size_t) (a * row + 4)] >= conf) idx.push_back(a);
    std::stable_sort(idx.begin(), idx.end(),
                     [&](int64_t a, int64_t b) { return host[(size_t) (a * row + 4)] > host[(size_t) (b * row + 4)]; });
    if (idx.size() > 30000) idx.resize(30000);
    std::vector<int64_t> kept;
    const auto area = [](const float* b) { return std::max(0.f, b[2] - b[0]) * std::max(0.f, b[3] - b[1]); };
    for (int64_t a : idx) {
        const float* b = &host[(size_t) (a * row)];
        bool ok = true;
        for (int64_t k : kept) {
            const float* c = &host[(size_t) (k * row)];
            const float iw2 = std::min(b[2], c[2]) - std::max(b[0], c[0]), ih2 = std::min(b[3], c[3]) - std::max(b[1], c[1]);
            if (iw2 <= 0 || ih2 <= 0) continue;
            const float inter = iw2 * ih2;
            if (inter / (area(b) + area(c) - inter) > iou) {
                ok = false;
                break;
            }
        }
        if (ok) kept.push_back(a);
        if ((int) kept.size() >= max_det) break;
    }

    // back to the picture: less the padding, by the scale, clipped
    const float gain = (float) r;
    std::vector<Detection> dets;
    for (int64_t a : kept) {
        const float* b = &host[(size_t) (a * row)];
        Detection d;
        d.x0 = std::clamp((b[0] - (float) left) / gain, 0.f, (float) W);
        d.y0 = std::clamp((b[1] - (float) top) / gain, 0.f, (float) H);
        d.x1 = std::clamp((b[2] - (float) left) / gain, 0.f, (float) W);
        d.y1 = std::clamp((b[3] - (float) top) / gain, 0.f, (float) H);
        d.score = b[4];
        d.cls = (int) b[5];
        dets.push_back(d);
    }

    if (masks != nullptr) {
        masks->clear();
        if (seg_ && !dets.empty()) {
            // the prototypes from the first level's map
            const Map& f = outs[(size_t) net.head_from[0]];
            const Map a = conv_map(net.p1, f);
            float* yu = net.pool.get<float>((size_t) (a.h * a.w * 4 * net.up_c));
            gemm_.f16(a.p, net.up_w, yu, a.h * a.w, 4 * net.up_c, a.c);
            Map u{net.pool.get<uint16_t>((size_t) (4 * a.h * a.w * net.up_c)), 2 * a.h, 2 * a.w, net.up_c};
            unshuffle2(yu, a.h, a.w, net.up_c, net.up_b, u.p);
            const Map pr = conv_map(net.p3, conv_map(net.p2, u));
            const int64_t N = (int64_t) dets.size(), Np = up8(N), HWp = pr.h * pr.w;
            std::vector<uint16_t> co((size_t) (Np * nm), 0);
            std::vector<float> bx((size_t) (N * 4));
            for (int64_t n = 0; n < N; ++n) {
                for (int k = 0; k < nm; ++k) co[(size_t) (n * nm + k)] = f16_from_f32(host[(size_t) (kept[(size_t) n] * row + 6 + k)]);
                bx[(size_t) (n * 4)] = dets[(size_t) n].x0;
                bx[(size_t) (n * 4 + 1)] = dets[(size_t) n].y0;
                bx[(size_t) (n * 4 + 2)] = dets[(size_t) n].x1;
                bx[(size_t) (n * 4 + 3)] = dets[(size_t) n].y1;
            }
            uint16_t* co_d = net.pool.get<uint16_t>(co.size());
            float* bx_d = net.pool.get<float>(bx.size());
            q.memcpy(co_d, co.data(), co.size() * 2);
            q.memcpy(bx_d, bx.data(), bx.size() * 4);
            float* lg = net.pool.get<float>((size_t) (HWp * Np));
            gemm_.f16(pr.p, co_d, lg, HWp, Np, nm);
            uint8_t* mk = net.pool.get<uint8_t>((size_t) (N * W * H));
            masks_kernel(lg, pr.h, pr.w, Np, N, bx_d, gain, (float) left, (float) top, (float) iw, (float) ih, W, H, mk);
            masks->assign((size_t) N, std::vector<uint8_t>((size_t) (W * H)));
            for (int64_t n = 0; n < N; ++n) q.memcpy((*masks)[(size_t) n].data(), mk + n * W * H, (size_t) (W * H));
            q.wait();
        }
    }
    q.wait();
    return dets;
}

}  // namespace strata::image
