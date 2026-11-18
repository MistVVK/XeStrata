// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/upscale/seedvr2_dit.cpp - see include/strata/upscale/seedvr2_dit.hpp.

#include "strata/upscale/seedvr2_dit.hpp"

#include "../image/dit_rows.hpp"
#include "../image/img_out.hpp"
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/platform/mapped_file.hpp"
#include "strata/image/qi_kernels.hpp"
#include "strata/image/safetensors.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/platform/env.hpp"

#include "strata/image/gpu_buf.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace strata::upscale {

// The kernels' helpers in a namespace of this file's own (SYCL names a lambda's kernel by its mangled type, and an
// anonymous namespace mangles the same in every file)
namespace seedvr2_dit_detail {

using image::DitOut;
namespace rows = image::dit_rows;
using image::q_of;
using image::alloc;
using image::patch_to_tokens;
using kernels::f16_gpu::f16_from_f32;
using kernels::f16_gpu::f32_from_f16;

constexpr float kNormEps = 1e-5f;   // the blocks' RMS norms and the heads' (the reference's norm_eps)
constexpr float kTimestep = 1000.f; // the one step's timestep (its sigma 1)
constexpr int32_t kPad = INT32_MIN; // a joint row past a window's rows
constexpr int64_t kTimeDim = 256;   // the timestep's sinusoid
constexpr int kTypeF16 = 1, kTypeQ8_0 = 8, kTypeW8 = -8;
constexpr int64_t kRotate = 256;     // int8_convrot's groups (its comfy_quant's convrot_groupsize)

// an int8_convrot matrix's rows (int8, one scale a row) as Q8_0 blocks: each block the row's scale and its 32 values
void int8_to_q8_0(const int8_t* w, const float* scale, int64_t n, int64_t k, uint8_t* out) {
    for (int64_t r = 0; r < n; ++r) {
        const uint16_t d = kernels::f16_from_f32(scale[r]);
        for (int64_t b = 0; b < k / 32; ++b) {
            uint8_t* o = out + (r * (k / 32) + b) * 34;
            std::memcpy(o, &d, 2);
            std::memcpy(o + 2, w + r * k + b * 32, 32);
        }
    }
}

// the patch embedder's input, in16 [N][kin]: token t's 2 x 2 positions j = 2 dy + dx of the 33 channels (the noise's
// 16, the condition's 16, a mask of ones) at j 33 + channel, the columns past 132 zero; the tokens' values are
// 4 c + j (MfVae's layout)
void patch_in16(const float* noise, const float* cond, int64_t N, int64_t lat, int64_t kin, uint16_t* in16, void* s) {
    const int64_t ch = 2 * lat + 1;
    q_of(s).parallel_for(sycl::range<1>((size_t) (N * kin)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t t = i / kin, k = i - t * kin;
        float v = 0.f;
        if (k < 4 * ch) {
            const int64_t j = k / ch, c = k - j * ch;
            v = c < lat ? noise[t * 4 * lat + 4 * c + j] : c < 2 * lat ? cond[t * 4 * lat + 4 * (c - lat) + j] : 1.f;
        }
        in16[i] = f16_from_f32(v);
    });
}

// the joint rows of the windows' attention: row r of q, k and v from the video's (src[r] >= 0, a token) or the text's
// (-1 - src[r]) rows, zero for kPad
void joint_gather(const int32_t* src, int64_t rows, int64_t C, const uint16_t* qv, const uint16_t* kv,
                  const uint16_t* vv, const uint16_t* qt, const uint16_t* kt, const uint16_t* vt, uint16_t* jq,
                  uint16_t* jk, uint16_t* jv, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (rows * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t r = i / C, c = i - r * C;
        const int32_t k = src[r];
        uint16_t a = 0, b = 0, d = 0;
        if (k >= 0) {
            const int64_t o = (int64_t) k * C + c;
            a = qv[o];
            b = kv[o];
            d = vv[o];
        } else if (k != kPad) {
            const int64_t o = (int64_t) (-1 - k) * C + c;
            a = qt[o];
            b = kt[o];
            d = vt[o];
        }
        jq[i] = a;
        jk[i] = b;
        jv[i] = d;
    });
}

// the windows' outputs back: the video rows to o16 [N][C] at their tokens, and tsum [L][C] the mean of the text rows'
// over the windows (starts [windows], each window's text rows first)
void joint_scatter(const int32_t* src, int64_t rows, int64_t C, const uint16_t* jo, uint16_t* o16, void* s) {
    q_of(s).parallel_for(sycl::range<1>((size_t) (rows * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t r = i / C, c = i - r * C;
        const int32_t k = src[r];
        if (k >= 0) o16[(int64_t) k * C + c] = jo[i];
    });
}

void text_mean(const int32_t* starts, int64_t windows, int64_t L, int64_t C, const uint16_t* jo, float* tsum,
               void* s) {
    const float inv = 1.f / (float) windows;
    q_of(s).parallel_for(sycl::range<1>((size_t) (L * C)), [=](sycl::id<1> id) {
        const auto i = static_cast<int64_t>(id[0]);
        const int64_t r = i / C, c = i - r * C;
        float v = 0.f;
        for (int64_t w = 0; w < windows; ++w) v += f32_from_f16(jo[((int64_t) starts[w] + r) * C + c]);
        tsum[i] = v * inv;
    });
}

// Python's round() of x (ties to even), as the reference's window sizes take it
int64_t py_round(double x) { return (int64_t) std::nearbyint(x); }

// a window: its first and last (exclusive) token row, column and frame
struct Win {
    int64_t t0, t1, h0, h1, w0, w1;
};

using Span = std::pair<int64_t, int64_t>;   // [first, end) along one axis

// One axis of `side` tokens cut into spans of `size`: back to back from 0, or, for the shifted layers, the same cut
// moved back by half a span (the first span its later half, the last one cut at the side).  The half rounds as the
// model's training layout rounds it: a span starts (size + 1) / 2 and ends size / 2 past its multiple of size.  An
// axis no longer than a span stays one span either way.
std::vector<Span> axis_spans(int64_t side, int64_t size, bool shifted) {
    std::vector<Span> out;
    if (!shifted || size >= side) {
        for (int64_t at = 0; at < side; at += size) out.emplace_back(at, std::min(at + size, side));
        return out;
    }
    const int64_t before = (size + 1) / 2, after = size / 2;
    for (int64_t k = 0, count = (side - 1) / size + 2; k < count; ++k) {
        const int64_t first = std::max<int64_t>(k * size - before, 0), end = std::min(k * size + after, side);
        if (end > first) out.emplace_back(first, end);
    }
    return out;
}

// The attention windows of a t x h x w token grid, as the model was trained on them: a 720p frame's grid (45 x 80
// tokens) and 30 frames split into num (frames, rows, columns) parts give a window's size, the grid scaled to 720p's
// area first; the windows tile the grid regular or shifted by half a window, the columns' spans outermost and the
// frames' innermost.
std::vector<Win> windows_of(int64_t t, int64_t h, int64_t w, bool shifted, const int64_t num[3]) {
    constexpr double kFrameTokens = 45.0 * 80.0;
    constexpr int64_t kClipFrames = 30;
    const double to720 = std::sqrt(kFrameTokens / (double) (h * w));
    const auto ceil_div = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
    const int64_t rows = ceil_div(py_round((double) h * to720), num[1]);
    const int64_t cols = ceil_div(py_round((double) w * to720), num[2]);
    const int64_t frames = ceil_div(std::min(t, kClipFrames), num[0]);
    const std::vector<Span> ts = axis_spans(t, frames, shifted), hs = axis_spans(h, rows, shifted),
                            ws = axis_spans(w, cols, shifted);
    std::vector<Win> out;
    out.reserve(ts.size() * hs.size() * ws.size());
    for (const Span& c : ws)
        for (const Span& r : hs)
            for (const Span& f : ts) out.push_back(Win{f.first, f.second, r.first, r.second, c.first, c.second});
    return out;
}

// torch.linspace(-1, 1, n)[i] in FP32, as rotary_embedding_torch's pixel positions
float pixel_pos(int64_t i, int64_t n) {
    if (n <= 1) return -1.f;
    const float step = 2.f / (float) (n - 1);
    return -1.f + (float) i * step;
}

}  // namespace seedvr2_dit_detail

using namespace seedvr2_dit_detail;

struct Seedvr2Dit::File {
    image::SafeTensors st;
    platform::MappedFile map;
    explicit File(const std::string& path) : st(path), map(path) {}
    const image::SafeTensor& get(const std::string& name) const {
        const image::SafeTensor* t = st.find(name);
        if (t == nullptr) throw std::runtime_error("image engine (SeedVR2 DiT): no " + name);
        return *t;
    }
    const uint8_t* raw(const image::SafeTensor& t) const {
        const uint64_t base = st.data_base();
        const uint64_t item = t.dtype == "F32" ? 4 : t.dtype == "F16" || t.dtype == "BF16" ? 2 :
                              t.dtype == "I8" || t.dtype == "U8" ? 1 : 0;
        uint64_t elements = 1;
        for (int64_t d : t.shape) {
            if (d < 0 || (d > 0 && elements > UINT64_MAX / (uint64_t) d))
                throw std::runtime_error("image engine (SeedVR2 DiT): invalid tensor shape");
            elements *= (uint64_t) d;
        }
        if (item == 0 || elements > UINT64_MAX / item || t.end < t.begin || base > map.size() ||
            t.end > map.size() - base || t.end - t.begin != elements * item)
            throw std::runtime_error("image engine (SeedVR2 DiT): invalid tensor data");
        return map.data() + base + t.begin;
    }
};

Seedvr2Dit::Seedvr2Dit() = default;

Seedvr2Dit::~Seedvr2Dit() {
    drop_activations();
    for (void* p : allocs_) gpu::free(p);
}

bool Seedvr2Dit::open(const std::string& path, void* stream, std::string& err, bool dry) {
    try {
        s_ = stream;
        dry_ = dry;
        file_ = std::make_unique<File>(path);
        File& f = *file_;
        const image::SafeTensor& vin = f.get("vid_in.proj.weight");
        C_ = vin.shape[0];
        lat_ = f.get("vid_out.proj.weight").shape[0] / 4;
        hd_ = f.get("blocks.0.attn.norm_q.vid.weight").shape[0];
        heads_ = C_ / hd_;
        F_ = f.get("blocks.0.mlp.vid.proj_in.weight").shape[0];
        // the 3B (dit_v2): a SwiGLU MLP, the blocks past the first few shared by video and text, the text rotated too,
        // and an RMS norm and modulation before the output
        v2_ = f.st.find("vid_out_norm.weight") != nullptr;
        if (v2_ != (f.st.find("blocks.0.mlp.vid.proj_in_gate.weight") != nullptr))
            throw std::runtime_error("image engine (SeedVR2 DiT): neither the 7B's blocks nor the 3B's");
        if (vin.shape[1] != 4 * (2 * lat_ + 1) || C_ % hd_ != 0 || hd_ % 8 != 0 || C_ % 8 != 0)
            throw std::runtime_error("image engine (SeedVR2 DiT): an unexpected shape");
        kin_ = (vin.shape[1] + 7) / 8 * 8;
        q8_ = f.get("blocks.0.attn.proj_qkv.vid.weight").dtype == "I8";
        if (q8_) {   // Comfy-Org's int8_convrot weights, their groups of 256 (the rotation RowsI8Of's kRegular makes)
            const image::SafeTensor& cq = f.get("blocks.0.attn.proj_qkv.vid.comfy_quant");
            const std::string meta(reinterpret_cast<const char*>(f.raw(cq)), (size_t) cq.elements());
            if (meta.find("int8_tensorwise") == std::string::npos || meta.find("\"convrot\": true") == std::string::npos ||
                meta.find("\"convrot_groupsize\": 256") == std::string::npos)
                throw std::runtime_error("image engine (SeedVR2 DiT): an 8-bit form it does not take: " + meta);
        }
        // W8A8 where it is not clearly the slower here, as every DiT chooses (dit_rows::w8_at_open): the int8 file's
        // rows as they are, the FP16 file's rotated and quantized on the GPU as int8_convrot's are.  Else the int8
        // file's rows as Q8_0, the FP16 file's as they are
        w8_ = !dry_ && C_ % kRotate == 0 && F_ % kRotate == 0 &&
              rows::w8_at_open(C_, v2_ ? 2 * F_ : F_, "SeedVR2's DiT", stream) == 1;
        {   // the RoPE: an axis's frequencies (the 7B's pixel ones, the 3B's lang ones), as the blocks' rope buffer holds
            // them; 3 axes of them in pairs
            const auto fr = f.st.f32("blocks.0.attn.rope.rope.freqs");
            freqs_ = fr;
            n_rot_ = (int64_t) fr.size() * 6;
            if (n_rot_ > hd_) throw std::runtime_error("image engine (SeedVR2 DiT): the RoPE is wider than a head");
        }
        // the parts outside the blocks, FP16 on the GPU (vid_in's columns padded to 8s)
        const auto dense = [&](const std::string& name, int64_t kpad, const float** bias) {
            const image::SafeTensor& t = f.get(name + ".weight");
            W w;
            w.type = kTypeF16;
            w.n = t.shape[0];
            const int64_t K = t.shape[1];
            w.k = std::max(kpad, K);
            weight_bytes_ += (uint64_t) (w.n * w.k) * 2 + (uint64_t) w.n * 4;
            if (dry_) return w;
            const auto v = f.st.f32(name + ".weight");
            std::vector<uint16_t> h((size_t) (w.n * w.k), 0);
            for (int64_t r = 0; r < w.n; ++r)
                for (int64_t c = 0; c < K; ++c) h[(size_t) (r * w.k + c)] = kernels::f16_from_f32(v[(size_t) (r * K + c)]);
            uint16_t* d = alloc<uint16_t>(allocs_, (int64_t) h.size(), "image engine (SeedVR2 DiT)");
            gpu::copy(d, h.data(), h.size() * 2);
            w.d = d;
            const auto b = f.st.f32(name + ".bias");
            float* db = alloc<float>(allocs_, (int64_t) b.size(), "image engine (SeedVR2 DiT)");
            gpu::copy(db, b.data(), b.size() * 4);
            *bias = db;
            return w;
        };
        vin_ = dense("vid_in.proj", kin_, &b_vin_);
        vout_ = dense("vid_out.proj", 0, &b_vout_);
        const image::SafeTensor& pc = f.get("positive_conditioning");
        L_ = pc.shape[0];
        // the time embedding of the one timestep (host, FP32): its sinusoid [sin, cos] (diffusers' get_timestep_embedding,
        // flip_sin_to_cos off), through proj_in, SiLU, proj_hid, SiLU, proj_out: [C][2 layers][3] (shift, scale, gate)
        std::vector<float> emb;
        {
            const auto lin = [&](const std::string& name, const std::vector<float>& x, bool silu) {
                const image::SafeTensor& t = f.get(name + ".weight");
                const int64_t n = t.shape[0], k = t.shape[1];
                const auto w = f.st.f32(name + ".weight"), b = f.st.f32(name + ".bias");
                std::vector<float> y((size_t) n);
                for (int64_t r = 0; r < n; ++r) {
                    double a = b[(size_t) r];
                    for (int64_t c = 0; c < k; ++c) a += (double) w[(size_t) (r * k + c)] * x[(size_t) c];
                    y[(size_t) r] = silu ? (float) (a / (1.0 + std::exp(-a))) : (float) a;
                }
                return y;
            };
            std::vector<float> sinus((size_t) kTimeDim);
            const int64_t half = kTimeDim / 2;
            for (int64_t i = 0; i < half; ++i) {
                const float e = std::exp(-std::log(10000.f) * (float) i / (float) half) * kTimestep;
                sinus[(size_t) i] = std::sin(e);
                sinus[(size_t) (half + i)] = std::cos(e);
            }
            emb = lin("emb_in.proj_out", lin("emb_in.proj_hid", lin("emb_in.proj_in", sinus, true), true), false);
            if ((int64_t) emb.size() != 6 * C_) throw std::runtime_error("image engine (SeedVR2 DiT): emb_in's output");
        }
        if (v2_) {   // the output's norm weight, and its modulation: the embedding's attn shift and scale plus vid_out_ada's
            const auto nw = f.st.f32("vid_out_norm.weight"), sc = f.st.f32("vid_out_ada.out_scale"),
                       sh = f.st.f32("vid_out_ada.out_shift");
            std::vector<float> v((size_t) (3 * C_));
            for (int64_t d = 0; d < C_; ++d) {
                v[(size_t) d] = nw[(size_t) d];
                v[(size_t) (C_ + d)] = emb[(size_t) ((d * 2) * 3 + 1)] + sc[(size_t) d];
                v[(size_t) (2 * C_ + d)] = emb[(size_t) ((d * 2) * 3)] + sh[(size_t) d];
            }
            weight_bytes_ += v.size() * 4;
            if (!dry_) {
                float* d = alloc<float>(allocs_, (int64_t) v.size(), "image engine (SeedVR2 DiT)");
                gpu::copy(d, v.data(), v.size() * 4);
                outmod_ = d;
            }
        }
        // the FP16 file's weights made W8A8 a matrix at a time through the activations' kernel (dit_rows::HalfI8 with
        // int8_convrot's rotation: the int8 file's rows, rounding apart), in scratch freed when the blocks are read.
        // (The 7B opens in 7.3 s so, 5.0 in FP16; quantized on the host's 16 threads it took 12.5: B70, i7-14700,
        // 2026-11-10)
        struct Scratch {
            std::vector<void*> p;
            ~Scratch() {
                for (void* q : p) gpu::free(q);
            }
        } qtmp;
        uint16_t* q16 = nullptr;
        int8_t* q8 = nullptr;
        float* qsc = nullptr;
        uint8_t* stage = nullptr;   // pinned: a pageable copy back took the 7B's 8 GB 5.6 s (B70, 2026-11-10)
        int64_t q_cap = 0, q_rows = 0;
        // the blocks: their FP16 weights from the file, and an FP32 region of their biases, head norms and modulations
        for (int i = 0;; ++i) {
            const std::string p = "blocks." + std::to_string(i) + ".";
            const bool shared = f.st.find(p + "attn.proj_qkv.all.weight") != nullptr;   // the 3B's later blocks
            if (!shared && f.st.find(p + "attn.proj_qkv.vid.weight") == nullptr) break;
            Block b;
            std::vector<image::LayerStream::Span> spans;
            std::vector<uint8_t> host;
            if (!host_.empty()) host.reserve(host_.back().size());   // the last block's: growing it copied 8 GB again
            std::vector<std::pair<uint64_t*, int>> fixups;   // (an offset to set, its span)
            std::vector<std::pair<W*, int>> wfix, fixups_w;   // a weight's span in the file, in the host region
            std::vector<std::pair<W*, int>> sfix;   // a W8A8 weight's scales' span
            std::vector<std::pair<int, int>> joined;   // spans that must follow each other (a fused weight's parts)
            // a weight of the file's rows of `names` one after another (the 3B's SwiGLU gate and input as one product)
            const auto weight = [&](W& w, const std::vector<std::string>& names) {
                std::vector<const image::SafeTensor*> ts, ss;
                w.n = 0;
                for (const std::string& name : names) {
                    const image::SafeTensor& t = f.get(name + ".weight");
                    ts.push_back(&t);
                    w.n += t.shape[0];
                    w.k = t.shape[1];
                }
                const auto join = [&](size_t first) {
                    for (size_t j = first + 1; j < spans.size(); ++j) joined.push_back({(int) j - 1, (int) j});
                };
                if (ts[0]->dtype == "F16" && !q8_ && !w8_) {
                    w.type = kTypeF16;
                    const size_t first = spans.size();
                    for (const auto* t : ts) spans.push_back({f.raw(*t), (uint64_t) (t->shape[0] * w.k) * 2, 0});
                    join(first);
                    wfix.push_back({&w, (int) first});
                    return;
                }
                if (ts[0]->dtype == "F16" && !q8_) {   // W8A8: the rows made int8 on the GPU, kept in the host region
                    w.type = kTypeW8;
                    const int64_t nk = w.n * w.k;
                    if (nk > q_cap || w.n > q_rows) {
                        gpu::stream_sync(s_);
                        for (void* p : qtmp.p) gpu::free(p);
                        qtmp.p.clear();
                        q16 = alloc<uint16_t>(qtmp.p, nk, "image engine (SeedVR2 DiT)");
                        q8 = alloc<int8_t>(qtmp.p, nk, "image engine (SeedVR2 DiT)");
                        qsc = alloc<float>(qtmp.p, w.n, "image engine (SeedVR2 DiT)");
                        stage = static_cast<uint8_t*>(gpu::alloc_host((size_t) nk * 2));
                        if (stage != nullptr) qtmp.p.push_back(stage);
                        q_cap = nk;
                        q_rows = w.n;
                    }
                    int64_t r0 = 0;
                    for (const auto* t : ts) {
                        if (t->dtype != "F16") throw std::runtime_error("image engine (SeedVR2 DiT): mixed dtypes");
                        const size_t bytes = (size_t) (t->shape[0] * w.k) * 2;
                        if (stage != nullptr) std::memcpy(stage + r0 * w.k * 2, f.raw(*t), bytes);
                        else gpu::copy_async(q16 + r0 * w.k, f.raw(*t), bytes, s_);
                        r0 += t->shape[0];
                    }
                    if (stage != nullptr) gpu::copy_async(q16, stage, (size_t) nk * 2, s_);
                    rows::rows_i8(rows::HalfI8<true>{q16, w.k}, w.n, w.k, q8, qsc, s_);
                    const size_t at = host.size();
                    host.resize(at + (size_t) nk + (size_t) w.n * 4);
                    uint8_t* back = stage != nullptr ? stage : host.data() + at;
                    gpu::copy_async(back, q8, (size_t) nk, s_);
                    gpu::copy_async(back + nk, qsc, (size_t) w.n * 4, s_);
                    if (!gpu::stream_sync(s_)) throw std::runtime_error("image engine (SeedVR2 DiT): the W8A8 rows");
                    if (stage != nullptr) std::memcpy(host.data() + at, stage, (size_t) nk + (size_t) w.n * 4);
                    spans.push_back({nullptr, (uint64_t) nk, (uint64_t) at});
                    fixups_w.push_back({&w, (int) spans.size() - 1});
                    spans.push_back({nullptr, (uint64_t) w.n * 4, (uint64_t) (at + nk)});
                    fixups.push_back({&w.sw, (int) spans.size() - 1});
                    return;
                }
                for (size_t j = 0; j < names.size(); ++j) {
                    const image::SafeTensor& sc = f.get(names[j] + ".weight_scale");
                    if (ts[j]->dtype != "I8" || sc.dtype != "F32" || sc.elements() != ts[j]->shape[0] ||
                        w.k % kRotate != 0)
                        throw std::runtime_error("image engine (SeedVR2 DiT): " + names[j] + " is not int8_convrot's");
                    ss.push_back(&sc);
                }
                if (w8_) {   // the file's rows and their scales as they are
                    w.type = kTypeW8;
                    size_t first = spans.size();
                    for (const auto* t : ts) spans.push_back({f.raw(*t), (uint64_t) (t->shape[0] * w.k), 0});
                    join(first);
                    wfix.push_back({&w, (int) first});
                    first = spans.size();
                    for (const auto* sc : ss) spans.push_back({f.raw(*sc), (uint64_t) sc->elements() * 4, 0});
                    join(first);
                    sfix.push_back({&w, (int) first});
                    return;
                }
                w.type = kTypeQ8_0;   // the same rows as Q8_0 blocks, made on the host
                scratch_ = std::max(scratch_, w.n * w.k);
                const size_t at = host.size(), bytes = (size_t) (w.n * (w.k / 32) * 34);
                if (!dry_) {   // (its sizes alone when dry)
                    host.resize(at + bytes);
                    int64_t r0 = 0;
                    for (size_t j = 0; j < ts.size(); ++j) {
                        const int64_t n = ts[j]->shape[0];
                        std::vector<float> scales((size_t) n);
                        std::memcpy(scales.data(), f.raw(*ss[j]), scales.size() * 4);
                        int8_to_q8_0(reinterpret_cast<const int8_t*>(f.raw(*ts[j])), scales.data(), n, w.k,
                                     host.data() + at + (size_t) (r0 * (w.k / 32) * 34));
                        r0 += n;
                    }
                }
                spans.push_back({nullptr, (uint64_t) bytes, (uint64_t) at});
                fixups_w.push_back({&w, (int) spans.size() - 1});
            };
            const auto vecf = [&](uint64_t& off, const std::vector<float>& v) {
                const size_t at = host.size();
                host.resize(at + v.size() * 4);
                std::memcpy(host.data() + at, v.data(), v.size() * 4);
                spans.push_back({nullptr, (uint64_t) v.size() * 4, (uint64_t) at});   // host set below
                fixups.push_back({&off, (int) spans.size() - 1});
            };
            for (int sd = 0; sd < (shared ? 1 : 2); ++sd) {
                Stream& st = sd == 0 ? b.v : b.t;
                const std::string tag = shared ? "all" : sd == 0 ? "vid" : "txt";
                const auto nm = [&](std::initializer_list<std::string_view> parts) {   // p + the parts
                    std::string n = p;
                    for (const std::string_view v : parts) n.append(v);
                    return n;
                };
                weight(st.qkv, {nm({"attn.proj_qkv.", tag})});
                weight(st.out, {nm({"attn.proj_out.", tag})});
                if (v2_) weight(st.in, {nm({"mlp.", tag, ".proj_in_gate"}), nm({"mlp.", tag, ".proj_in"})});
                else weight(st.in, {nm({"mlp.", tag, ".proj_in"})});
                weight(st.fc, {nm({"mlp.", tag, ".proj_out"})});
                vecf(st.b_out, f.st.f32(nm({"attn.proj_out.", tag, ".bias"})));
                if (!v2_) {   // the 3B's MLP has no biases
                    vecf(st.b_in, f.st.f32(nm({"mlp.", tag, ".proj_in.bias"})));
                    vecf(st.b_fc, f.st.f32(nm({"mlp.", tag, ".proj_out.bias"})));
                }
                vecf(st.nq, f.st.f32(nm({"attn.norm_q.", tag, ".weight"})));
                vecf(st.nk, f.st.f32(nm({"attn.norm_k.", tag, ".weight"})));
                // the modulation: the embedding's (shift, scale, gate) of a layer at [d][layer][3] plus the block's own
                std::vector<float> mod((size_t) (6 * C_));
                const char* names[2] = {"attn", "mlp"};
                for (int l = 0; l < 2; ++l) {
                    const auto sh = f.st.f32(nm({"ada.", tag, ".", names[l], "_shift"}));
                    const auto sc = f.st.f32(nm({"ada.", tag, ".", names[l], "_scale"}));
                    const auto gt = f.st.f32(nm({"ada.", tag, ".", names[l], "_gate"}));
                    for (int64_t d = 0; d < C_; ++d) {
                        const float* e = emb.data() + (d * 2 + l) * 3;
                        mod[(size_t) ((l * 3 + 0) * C_ + d)] = e[0] + sh[(size_t) d];
                        mod[(size_t) ((l * 3 + 1) * C_ + d)] = e[1] + sc[(size_t) d];
                        mod[(size_t) ((l * 3 + 2) * C_ + d)] = e[2] + gt[(size_t) d];
                    }
                }
                vecf(st.mod, mod);
            }
            host_.push_back(std::move(host));
            for (auto& [off, k] : fixups) spans[(size_t) k].host = host_.back().data() + spans[(size_t) k].offset;
            for (auto& [w, k] : fixups_w) spans[(size_t) k].host = host_.back().data() + spans[(size_t) k].offset;
            for (auto& sp : spans) sp.offset = 0;   // the stream assigns them
            const int layer = blocks_.add(spans);
            for (auto& [w, k] : wfix) w->d = reinterpret_cast<const void*>(blocks_.offset(layer, k));
            for (auto& [w, k] : fixups_w) w->d = reinterpret_cast<const void*>(blocks_.offset(layer, k));
            for (auto& [w, k] : sfix) w->sw = blocks_.offset(layer, k);
            for (auto& [off, k] : fixups) *off = blocks_.offset(layer, k);
            for (auto& [a, c] : joined)
                if (blocks_.offset(layer, c) != blocks_.offset(layer, a) + spans[(size_t) a].bytes)
                    throw std::runtime_error("image engine (SeedVR2 DiT): a fused weight's parts not side by side");
            if (shared) b.t = b.v;   // one set of weights for both streams (their offsets set above)
            blocks_v_.push_back(b);
        }
        if (blocks_v_.empty()) throw std::runtime_error("image engine (SeedVR2 DiT): no blocks");
        if (blocks_v_[0].v.qkv.n != 3 * C_ || blocks_v_[0].v.in.n != (v2_ ? 2 * F_ : F_))
            throw std::runtime_error("image engine (SeedVR2 DiT): an unexpected block shape");
        if (dry_) return true;
        if (!gemm_.init(s_, scratch_, err)) return false;
        {   // until the caller sets the budget (set_block_budget): the blocks that fit half the free VRAM
            size_t free = 0, total = 0;
            gpu::mem_info(&free, &total);
            blocks_.configure(std::max(blocks_.min_budget(), std::min<uint64_t>(blocks_.total(), free / 2)), s_);
        }
        {   // the text through txt_in, once: its rows the blocks' text stream starts from
            std::vector<void*> tmp;
            const float* b_tin = nullptr;
            const W tin = dense("txt_in", 0, &b_tin);
            const auto pv = f.st.f32("positive_conditioning");
            std::vector<uint16_t> h(pv.size());
            for (size_t i = 0; i < pv.size(); ++i) h[i] = kernels::f16_from_f32(pv[i]);
            uint16_t* p16 = alloc<uint16_t>(tmp, (int64_t) h.size(), "image engine (SeedVR2 DiT)");
            gpu::copy(p16, h.data(), h.size() * 2);
            txt0_ = alloc<float>(allocs_, L_ * C_, "image engine (SeedVR2 DiT)");
            product(p16, nullptr, tin, nullptr, txt0_, L_, b_tin);
            gpu::stream_sync(s_);
            for (void* q : tmp) gpu::free(q);
            // txt_in's weights are not needed again
            for (const void* q : {tin.d, (const void*) b_tin}) {
                auto it = std::find(allocs_.begin(), allocs_.end(), q);
                if (it != allocs_.end()) {
                    gpu::free(*it);
                    allocs_.erase(it);
                }
            }
            weight_bytes_ -= (uint64_t) (tin.n * tin.k) * 2 + (uint64_t) tin.n * 4;
            weight_bytes_ += (uint64_t) (L_ * C_) * 4;
        }
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

// the window layouts of a t x h x w grid (t latent frames): each window's joint rows (its text rows, then its video
// tokens in the window's order, padded to qi_attn_rows) and its tokens' RoPE, window-local
void Seedvr2Dit::layout(int which, int64_t t, int64_t h, int64_t w, std::vector<void*>& owned) {
    static const int64_t kNum[3] = {4, 3, 3};   // the release's window: 720p's grid in 4 x 3 x 3
    const std::vector<Win> wins = windows_of(t, h, w, which == 1, kNum);
    Layout& l = lay_[which];
    l.windows = (int64_t) wins.size();
    l.at.clear();
    l.len.clear();
    std::vector<int32_t> src;
    const int64_t half = hd_ / 2, nf = (int64_t) freqs_.size();
    const int64_t N = t * h * w;
    std::vector<float> cs((size_t) (N * half), 1.f), sn((size_t) (N * half), 0.f);
    int64_t covered = 0;
    for (const Win& v : wins) {
        const int64_t wt = v.t1 - v.t0, wh = v.h1 - v.h0, ww = v.w1 - v.w0, n = wt * wh * ww, rows = L_ + n;
        l.at.push_back((int64_t) src.size());
        l.len.push_back(rows);
        for (int64_t r = 0; r < L_; ++r) src.push_back((int32_t) (-1 - r));
        for (int64_t a = 0; a < wt; ++a)
            for (int64_t y = 0; y < wh; ++y)
                for (int64_t x = 0; x < ww; ++x) {
                    const int64_t tok = ((v.t0 + a) * h + v.h0 + y) * w + (v.w0 + x);
                    src.push_back((int32_t) tok);
                    // the 7B's pixel positions (-1 .. 1 over the window), the 3B's integer ones (the frame after the
                    // text's rows)
                    const float pos[3] = {v2_ ? (float) (L_ + a) : pixel_pos(a, wt), v2_ ? (float) y : pixel_pos(y, wh),
                                          v2_ ? (float) x : pixel_pos(x, ww)};
                    for (int axis = 0; axis < 3; ++axis)
                        for (int64_t j = 0; j < nf; ++j) {
                            const float ang = pos[axis] * freqs_[(size_t) j];
                            cs[(size_t) (tok * half + axis * nf + j)] = std::cos(ang);
                            sn[(size_t) (tok * half + axis * nf + j)] = std::sin(ang);
                        }
                    ++covered;
                }
        for (int64_t r = rows; r < image::qi_attn_rows(rows); ++r) src.push_back(kPad);
    }
    if (covered != N) throw std::runtime_error("image engine (SeedVR2 DiT): the windows are not a partition");
    l.jrows = (int64_t) src.size();
    l.src = alloc<int32_t>(owned, l.jrows, "image engine (SeedVR2 DiT)");
    gpu::copy(l.src, src.data(), src.size() * 4);
    const std::vector<int32_t> starts(l.at.begin(), l.at.end());
    l.starts = alloc<int32_t>(owned, l.windows, "image engine (SeedVR2 DiT)");
    gpu::copy(l.starts, starts.data(), starts.size() * 4);
    l.cos = alloc<float>(owned, N * half, "image engine (SeedVR2 DiT)");
    l.sin = alloc<float>(owned, N * half, "image engine (SeedVR2 DiT)");
    gpu::copy(l.cos, cs.data(), cs.size() * 4);
    gpu::copy(l.sin, sn.data(), sn.size() * 4);
}

uint64_t Seedvr2Dit::activation_bytes(int64_t h, int64_t w, int64_t t) const {
    const int64_t N = t * h * w;
    // the joint rows at most: each window's text, its tokens, and up to 127 rows of padding a window
    int64_t jr = 0;
    for (int which = 0; which < 2; ++which) {
        static const int64_t kNum[3] = {4, 3, 3};
        const auto wins = windows_of(t, h, w, which == 1, kNum);
        int64_t r = 0;
        for (const Win& v : wins) r += image::qi_attn_rows(L_ + (v.t1 - v.t0) * (v.h1 - v.h0) * (v.w1 - v.w0));
        jr = std::max(jr, r);
    }
    const uint64_t f32 = (uint64_t) ((N + std::max(N, L_)) * C_ + L_ * C_ * 2 + N * 3 * C_ + L_ * 3 * C_ + std::max(N, L_) * (v2_ ? 2 : 1) * F_ +
                                     2 * std::max(N, L_) + (v2_ ? L_ * hd_ : 0));
    const uint64_t f16 = (uint64_t) (N * kin_ + std::max(N, L_) * C_ + 3 * (N + L_) * C_ + 4 * jr * C_ + N * C_ +
                                     std::max(N, L_) * F_);
    const uint64_t tables = (uint64_t) (N * hd_ * 2) * 4 + (uint64_t) (jr * 2) * 4;
    const uint64_t rows8 = w8_ ? (uint64_t) (std::max(N, L_) * std::max(C_, F_)) + (uint64_t) std::max(N, L_) * 8
                           : q8_ ? (uint64_t) (std::max(N, L_) * std::max(C_, F_)) * 2 : 0;
    return f32 * 4 + f16 * 2 + tables + rows8;
}

void Seedvr2Dit::drop_activations() {
    for (void* p : act_) gpu::free(p);
    act_.clear();
    cap_t_ = cap_h_ = cap_w_ = 0;
    held_ = 0;
    for (Layout& l : lay_) l = Layout();
}

void Seedvr2Dit::ensure(int64_t t, int64_t h, int64_t w) {
    if (t == cap_t_ && h == cap_h_ && w == cap_w_) return;
    drop_activations();
    const int64_t N = t * h * w;
    layout(0, t, h, w, act_);
    layout(1, t, h, w, act_);
    const int64_t jr = std::max(lay_[0].jrows, lay_[1].jrows);
    x_ = alloc<float>(act_, N * C_, "image engine (SeedVR2 DiT)");
    y_ = alloc<float>(act_, std::max(N, L_) * C_, "image engine (SeedVR2 DiT)");
    t_ = alloc<float>(act_, L_ * C_, "image engine (SeedVR2 DiT)");
    tsum_ = alloc<float>(act_, L_ * C_, "image engine (SeedVR2 DiT)");
    qkv_ = alloc<float>(act_, N * 3 * C_, "image engine (SeedVR2 DiT)");
    qkt_ = alloc<float>(act_, L_ * 3 * C_, "image engine (SeedVR2 DiT)");
    hid_ = alloc<float>(act_, std::max(N, L_) * (v2_ ? 2 * F_ : F_), "image engine (SeedVR2 DiT)");
    if (v2_) {   // the text's RoPE: row r on all three axes
        const int64_t half = hd_ / 2, nf = (int64_t) freqs_.size();
        std::vector<float> cs((size_t) (L_ * half), 1.f), sn((size_t) (L_ * half), 0.f);
        for (int64_t r = 0; r < L_; ++r)
            for (int axis = 0; axis < 3; ++axis)
                for (int64_t j = 0; j < nf; ++j) {
                    const float ang = (float) r * freqs_[(size_t) j];
                    cs[(size_t) (r * half + axis * nf + j)] = std::cos(ang);
                    sn[(size_t) (r * half + axis * nf + j)] = std::sin(ang);
                }
        tcos_ = alloc<float>(act_, L_ * half, "image engine (SeedVR2 DiT)");
        tsin_ = alloc<float>(act_, L_ * half, "image engine (SeedVR2 DiT)");
        gpu::copy(tcos_, cs.data(), cs.size() * 4);
        gpu::copy(tsin_, sn.data(), sn.size() * 4);
    }
    rs_ = alloc<float>(act_, std::max(N, L_), "image engine (SeedVR2 DiT)");
    rs2_ = alloc<float>(act_, std::max(N, L_), "image engine (SeedVR2 DiT)");
    in16_ = alloc<uint16_t>(act_, N * kin_, "image engine (SeedVR2 DiT)");
    a16_ = alloc<uint16_t>(act_, std::max(N, L_) * C_, "image engine (SeedVR2 DiT)");
    q16_ = alloc<uint16_t>(act_, (N + L_) * C_, "image engine (SeedVR2 DiT)");
    k16_ = alloc<uint16_t>(act_, (N + L_) * C_, "image engine (SeedVR2 DiT)");
    v16_ = alloc<uint16_t>(act_, (N + L_) * C_, "image engine (SeedVR2 DiT)");
    jq_ = alloc<uint16_t>(act_, jr * C_, "image engine (SeedVR2 DiT)");
    jk_ = alloc<uint16_t>(act_, jr * C_, "image engine (SeedVR2 DiT)");
    jv_ = alloc<uint16_t>(act_, jr * C_, "image engine (SeedVR2 DiT)");
    jo_ = alloc<uint16_t>(act_, jr * C_, "image engine (SeedVR2 DiT)");
    o16_ = alloc<uint16_t>(act_, N * C_, "image engine (SeedVR2 DiT)");
    g16_ = alloc<uint16_t>(act_, std::max(N, L_) * F_, "image engine (SeedVR2 DiT)");
    if (w8_) {
        xq8_ = alloc<int8_t>(act_, std::max(N, L_) * std::max(C_, F_), "image engine (SeedVR2 DiT)");
        xs_ = alloc<float>(act_, std::max(N, L_), "image engine (SeedVR2 DiT)");
        xs2_ = alloc<float>(act_, std::max(N, L_), "image engine (SeedVR2 DiT)");
    } else if (q8_) {
        rot16_ = alloc<uint16_t>(act_, std::max(N, L_) * std::max(C_, F_), "image engine (SeedVR2 DiT)");
    }
    cap_t_ = t;
    cap_h_ = h;
    cap_w_ = w;
    held_ = activation_bytes(h, w, t);
}

image::DitProduct Seedvr2Dit::operands(const uint16_t* x16, const float* rs, const W& w, const uint8_t* base,
                                       float* y, int64_t T, const float* bias) const {
    image::DitProduct p;
    p.w = &w;
    p.d = base != nullptr ? base + reinterpret_cast<uintptr_t>(w.d) : static_cast<const uint8_t*>(w.d);
    p.x16 = x16;
    if (w.type == kTypeW8) {   // input's int8 rows (xq8_ and their scales xs_in_) against the file's
        // its scales an offset in the block's bytes: an int8 weight comes with its base (dense() makes FP16 ones)
        if (base == nullptr) throw std::logic_error("image engine (SeedVR2 DiT): an int8 weight without its block");
        p.x8 = xq8_;
        p.sx = xs_in_;
        p.w8 = reinterpret_cast<const int8_t*>(p.d);
        p.sw = reinterpret_cast<const float*>(base + w.sw);
    }
    p.rs = rs;
    p.bias = bias;
    p.y = y;
    p.T = T;
    return p;
}

DitOut Seedvr2Dit::product_out(const uint16_t* x16, const float* rs, const W& w, const uint8_t* base, float* y,
                               int64_t T, const float* bias) {
    image::DitProduct p = operands(x16, rs, w, base, y, T, bias);
    p.defer = true;
    return prod_.run(p);
}

void Seedvr2Dit::product(const uint16_t* x16, const float* rs, const W& w, const uint8_t* base, float* y, int64_t T,
                         const float* bias) {
    prod_.run(operands(x16, rs, w, base, y, T, bias));
}

float* Seedvr2Dit::next_xs() {
    xs_in_ = xs_in_ == xs_ ? xs2_ : xs_;
    return xs_in_;
}

template<class Op16, class OpI8>
const uint16_t* Seedvr2Dit::input(const Op16& op16, const OpI8& opi8, int64_t T, int64_t K, uint16_t* y16,
                                  float* rs) {
    if (w8_) {
        rows::rows_i8(opi8, T, K, xq8_, next_xs(), s_);
        return nullptr;
    }
    rows::rows16(op16, T, K, y16, rs, s_);
    if (!q8_) return y16;
    rows::regular_rotate16(y16, T, K, rot16_, s_);   // the Q8_0 rows' basis
    return rot16_;
}

// a stream's attention input: RMSNorm(x) * scale + shift (the block's attn modulation), its q, k, v product
DitOut Seedvr2Dit::stream_attn_in(const Stream& st, const uint8_t* base, const float* x, int64_t T, float* qkv) {
    const float* mod = reinterpret_cast<const float*>(base + st.mod);
    const uint16_t* in = input(rows::RmsAffine{x, nullptr, mod + C_, mod, C_, kNormEps},
                               rows::RmsAffineI8<true>{x, mod + C_, mod, C_, kNormEps}, T, C_, a16_, rs_);
    return product_out(in, rs_, st.qkv, base, qkv, T, nullptr);
}

// x += gate * proj_out(o16) (o16: the attention's output, FP16 rows)
void Seedvr2Dit::stream_out(const Stream& st, const uint8_t* base, const uint16_t* o16, float* x, int64_t T) {
    const float* mod = reinterpret_cast<const float*>(base + st.mod);
    const uint16_t* in = o16;
    if (w8_) {
        rows::rows_i8(rows::HalfI8<true>{o16, C_}, T, C_, xq8_, next_xs(), s_);
    } else if (q8_) {
        rows::regular_rotate16(o16, T, C_, rot16_, s_);
        in = rot16_;
    }
    rows::gate_add(x, product_out(in, nullptr, st.out, base, y_, T, reinterpret_cast<const float*>(base + st.b_out)),
                   mod + 2 * C_, T, C_, s_);
}

// x += gate * MLP(RMSNorm(x) * scale + shift), the MLP's GELU (tanh) between its two products
void Seedvr2Dit::stream_mlp(const Stream& st, const uint8_t* base, float* x, int64_t T) {
    const float* mod = reinterpret_cast<const float*>(base + st.mod);
    const uint16_t* in = input(rows::RmsAffine{x, nullptr, mod + 4 * C_, mod + 3 * C_, C_, kNormEps},
                               rows::RmsAffineI8<true>{x, mod + 4 * C_, mod + 3 * C_, C_, kNormEps}, T, C_, a16_, rs_);
    DitOut y;
    if (v2_) {   // the 3B's SwiGLU: the gate's and the input's product as one, no biases
        const DitOut g = product_out(in, rs_, st.in, base, hid_, T, nullptr);
        in = input(rows::SwiGlu{g, F_}, rows::SwiGluI8<true>{g, F_}, T, F_, g16_, rs2_);
        y = product_out(in, rs2_, st.fc, base, y_, T, nullptr);
    } else {
        const DitOut hid = product_out(in, rs_, st.in, base, hid_, T, reinterpret_cast<const float*>(base + st.b_in));
        in = input(rows::GeluTanh{hid}, rows::GeluTanhI8<true>{hid}, T, F_, g16_, rs2_);
        y = product_out(in, rs2_, st.fc, base, y_, T, reinterpret_cast<const float*>(base + st.b_fc));
    }
    rows::gate_add(x, y, mod + 5 * C_, T, C_, s_);
}

void Seedvr2Dit::forward(const float* noise, const float* cond, int64_t h, int64_t w, float* out, int64_t t,
                         const std::function<void(int, int)>& block) {
    ensure(t, h, w);
    const int64_t N = t * h * w, C = C_, L = L_;
    patch_in16(noise, cond, N, lat_, kin_, in16_, s_);
    product(in16_, nullptr, vin_, nullptr, x_, N, b_vin_);
    gpu::copy_async(t_, txt0_, (size_t) (L * C) * 4, s_);
    const int nb = (int) blocks_v_.size();
    for (int i = 0; i < nb; ++i) {
        const Block& b = blocks_v_[(size_t) i];
        const uint8_t* base = blocks_.begin(i);
        const Layout& l = lay_[i % 2];
        const bool last = i == nb - 1;
        // q, k, v of both streams: the video's rotated by its tokens' window-local RoPE, the text's not.  A product is
        // read as it stands (its rows' scales and bias applied by its reader), so the video's is read before the
        // text's input takes the scales' buffer
        const auto part = [&](const DitOut& o, int64_t k) {
            DitOut p = o;
            if (p.f != nullptr) p.f += k * C;
            if (p.i != nullptr) p.i += k * C;
            if (p.bf != nullptr) p.bf += k * C;
            if (p.sw != nullptr) p.sw += k * C;
            if (p.b != nullptr) p.b += k * C;
            return p;
        };
        const DitOut qv = stream_attn_in(b.v, base, x_, N, qkv_);
        rows::qk_rope16(part(qv, 0), part(qv, 1), part(qv, 2), reinterpret_cast<const float*>(base + b.v.nq),
                        reinterpret_cast<const float*>(base + b.v.nk), l.cos, l.sin, nullptr, N, heads_, hd_, false,
                        q16_, k16_, v16_, s_, kNormEps, n_rot_);
        const DitOut qt = stream_attn_in(b.t, base, t_, L, qkt_);
        rows::qk_rope16(part(qt, 0), part(qt, 1), part(qt, 2), reinterpret_cast<const float*>(base + b.t.nq),
                        reinterpret_cast<const float*>(base + b.t.nk), tcos_, tsin_, nullptr, L, heads_, hd_,
                        false, q16_ + N * C, k16_ + N * C, v16_ + N * C, s_, kNormEps, n_rot_);
        // the windows' joint attention, each window's rows on their own
        joint_gather(l.src, l.jrows, C, q16_, k16_, v16_, q16_ + N * C, k16_ + N * C, v16_ + N * C, jq_, jk_, jv_,
                     s_);
        for (int64_t k = 0; k < l.windows; ++k) {
            const int64_t a = l.at[(size_t) k];
            image::qi_attention(jq_ + a * C, jk_ + a * C, jv_ + a * C, l.len[(size_t) k], heads_, hd_, 0,
                                jo_ + a * C, s_, nullptr, 0, true);
        }
        joint_scatter(l.src, l.jrows, C, jo_, o16_, s_);
        stream_out(b.v, base, o16_, x_, N);
        stream_mlp(b.v, base, x_, N);
        if (!last) {   // the text: its windows' mean, then as the video
            text_mean(l.starts, l.windows, L, C, jo_, tsum_, s_);
            kernels::f32_to_f16_bulk(tsum_, a16_, L * C, s_);
            stream_out(b.t, base, a16_, t_, L);
            stream_mlp(b.t, base, t_, L);
        }
        blocks_.end(i);
        if (block) {   // told once the GPU has run it, not when it is queued
            gpu::stream_sync(s_);
            block(i + 1, nb);
        }
        if (const char* dump = strata::env_get("SEEDVR2_DUMP")) {   // a check: the video stream after each block
            gpu::stream_sync(s_);
            std::vector<float> hx((size_t) (N * C));
            gpu::copy(hx.data(), x_, hx.size() * 4);
            if (FILE* fo = std::fopen((std::string(dump) + "/x" + std::to_string(i) + ".f32").c_str(), "wb")) {
                std::fwrite(hx.data(), 4, hx.size(), fo);
                std::fclose(fo);
            }
            std::vector<float> ht((size_t) (L * C));
            gpu::copy(ht.data(), t_, ht.size() * 4);
            if (FILE* fo = std::fopen((std::string(dump) + "/t" + std::to_string(i) + ".f32").c_str(), "wb")) {
                std::fwrite(ht.data(), 4, ht.size(), fo);
                std::fclose(fo);
            }
        }
    }
    // the output projection (the 3B's after its norm and modulation), back to the tokens' order
    if (v2_) {
        rows::rows16(rows::RmsAffine{x_, outmod_, outmod_ + C, outmod_ + 2 * C, C, kNormEps}, N, C, a16_, rs_, s_);
        product(a16_, rs_, vout_, nullptr, y_, N, b_vout_);
    } else {
        kernels::f32_to_f16_bulk(x_, a16_, N * C, s_);
        product(a16_, nullptr, vout_, nullptr, y_, N, b_vout_);
    }
    patch_to_tokens(y_, N, lat_, out, s_);
    gpu::stream_sync(s_);
}

}  // namespace strata::upscale
