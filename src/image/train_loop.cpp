// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/train_loop.cpp - see include/strata/image/train_loop.hpp.
#include "strata/image/train_loop.hpp"

#include "strata/core/gpu.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include "dit_rows.hpp"

namespace strata::image::train {

namespace {

constexpr int64_t kSteps = 1000;   // num_train_timesteps

// np.linspace(sigma_max * 1000, sigma_min * 1000, 1000) / 1000 with the scheduler's sigma_max 1 and sigma_min 0.001
std::vector<double> base_sigmas() {
    std::vector<double> s((size_t) kSteps);
    for (int64_t i = 0; i < kSteps; ++i) s[(size_t) i] = (1000.0 + (1.0 - 1000.0) * (double) i / (kSteps - 1)) / 1000.0;
    return s;
}

// diffusers' stretch_shift_to_terminal, then the FP32 sigmas times 1000 in FP32 (torch's timesteps)
Schedule finish(std::vector<double> s, double terminal) {
    if (terminal > 0.0) {
        const double scale = (1.0 - s.back()) / (1.0 - terminal);
        for (double& v : s) v = 1.0 - (1.0 - v) / scale;
    }
    Schedule out;
    out.t.resize(s.size());
    for (size_t i = 0; i < s.size(); ++i) out.t[i] = (float) s[i] * 1000.0f;
    return out;
}

}  // namespace

Schedule schedule_linear() {
    Schedule out;
    out.t.resize((size_t) kSteps);
    // torch.linspace(1000, 1, 1000) in FP32: start + i * step, the step (1 - 1000) / 999
    const float step = (1.0f - 1000.0f) / (float) (kSteps - 1);
    for (int64_t i = 0; i < kSteps; ++i)
        out.t[(size_t) i] = i < kSteps / 2 ? 1000.0f + step * (float) i : 1.0f - step * (float) (kSteps - 1 - i);
    return out;
}

Schedule schedule_shift(double shift) {
    std::vector<double> s = base_sigmas();
    for (double& v : s) v = shift * v / (1.0 + (shift - 1.0) * v);
    return finish(s, 0.0);
}

Schedule schedule_dynamic(int64_t image_seq_len, double base_seq, double max_seq, double base_shift, double max_shift,
                          double terminal) {
    const double m = (max_shift - base_shift) / (max_seq - base_seq), b = base_shift - m * base_seq;
    const double mu = (double) image_seq_len * m + b, e = std::exp(mu);
    std::vector<double> s = base_sigmas();
    for (double& v : s) v = e / (e + (1.0 / v - 1.0));
    return finish(s, terminal);
}

bool parse_index_draw(const std::string& s, IndexDraw& out) {
    if (s == "balanced") out = IndexDraw::balanced;
    else if (s == "content") out = IndexDraw::content;
    else if (s == "style") out = IndexDraw::style;
    else return false;
    return true;
}

int64_t draw_index(std::mt19937_64& rng, IndexDraw how) {
    if (how == IndexDraw::balanced) return std::uniform_int_distribution<int64_t>(0, kSteps - 2)(rng);   // randint(0, 999)
    const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng), c = u * u * u;
    const double x = (how == IndexDraw::content ? c : 1.0 - c) * (double) kSteps;
    return std::clamp<int64_t>((int64_t) x, 0, kSteps - 1);
}

void lora_init(std::mt19937_64& rng, int64_t rank, int64_t K, std::vector<float>& A) {
    const float bound = 1.0f / std::sqrt((float) K);
    std::uniform_real_distribution<float> u(-bound, bound);
    A.resize((size_t) (rank * K));
    for (float& v : A) v = u(rng);
}

double sq_sum(const std::vector<std::pair<const float*, int64_t>>& parts, void* s) {
    constexpr int64_t kWG = 256, kGroups = 64;   // a part's work-groups: each writes one partial sum
    const size_t n_part = parts.size() * kGroups;
    static std::mutex mu;
    static std::map<void*, std::pair<float*, size_t>> kept;   // the partials' buffer, kept for the stream
    std::lock_guard<std::mutex> lock(mu);
    auto& [part, cap] = kept[s];
    if (cap < n_part) {
        if (part != nullptr) {
            gpu::stream_sync(s);
            gpu::free(part);
        }
        part = nullptr;
        if (!gpu::alloc_device(&part, n_part * 4)) throw std::runtime_error("image engine: the norm's sums");
        cap = n_part;
    }
    for (size_t k = 0; k < parts.size(); ++k) {
        const float* g = parts[k].first;
        const int64_t n = parts[k].second;
        float* out = part + k * kGroups;
        dit_rows::q_of(s).parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) (kGroups * kWG)), sycl::range<1>((size_t) kWG)),
            [=](sycl::nd_item<1> it) {
                float v = 0.f;
                for (int64_t i = (int64_t) it.get_global_linear_id(); i < n; i += kGroups * kWG) v += g[i] * g[i];
                v = sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>());
                if (it.get_local_linear_id() == 0) out[it.get_group_linear_id()] = v;
            });
    }
    std::vector<float> h(n_part);
    gpu::stream_sync(s);
    if (!gpu::copy(h.data(), part, n_part * 4)) throw std::runtime_error("image engine: the norm's sums read back");
    double t = 0.0;
    for (float v : h) t += v;
    return t;
}

void adamw(float* w, float* g, float* m, float* v, int64_t n, const AdamW& o, int64_t step, float gscale, void* s) {
    const float b1 = o.beta1, b2 = o.beta2, lr = o.lr, eps = o.eps, wd = o.wd;
    const float bc1 = 1.f - std::pow(b1, (float) step), bc2 = 1.f - std::pow(b2, (float) step);
    dit_rows::q_of(s).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const float gg = g[i] * gscale;
        const float nm = m[i] = b1 * m[i] + (1.f - b1) * gg;
        const float nv = v[i] = b2 * v[i] + (1.f - b2) * gg * gg;
        const float p = w[i] * (1.f - lr * wd);   // torch's AdamW: the decay first, on the old value
        w[i] = p - (lr / bc1) * nm / (sycl::sqrt(nv) / sycl::sqrt(bc2) + eps);
        g[i] = 0.f;
    });
}

void save_safetensors_bf16(const std::string& path, std::vector<Tensor> ts,
                           const std::vector<std::pair<std::string, std::string>>& meta) {
    std::sort(ts.begin(), ts.end(), [](const Tensor& a, const Tensor& b) { return a.key < b.key; });
    auto esc = [](const std::string& x) {
        std::string o;
        for (char c : x) {
            if (c == '"' || c == '\\') o += '\\';
            o += c;
        }
        return o;
    };
    std::ostringstream head;
    head << "{\"__metadata__\":{";
    for (size_t i = 0; i < meta.size(); ++i)
        head << (i ? "," : "") << "\"" << esc(meta[i].first) << "\":\"" << esc(meta[i].second) << "\"";
    head << "}";
    int64_t off = 0;
    for (const Tensor& t : ts) {
        head << ",\"" << esc(t.key) << "\":{\"dtype\":\"BF16\",\"shape\":[";
        for (size_t i = 0; i < t.shape.size(); ++i) head << (i ? "," : "") << t.shape[i];
        const int64_t bytes = (int64_t) t.v.size() * 2;
        head << "],\"data_offsets\":[" << off << "," << off + bytes << "]}";
        off += bytes;
    }
    head << "}";
    std::string hs = head.str();
    hs.append((8 - hs.size() % 8) % 8, ' ');   // the data starts on 8 bytes, as safetensors' own writer pads it
    const std::string tmp = path + ".part";
    {
        std::ofstream f(tmp, std::ios::binary);
        if (!f) throw std::runtime_error("image engine: cannot write " + tmp);
        const uint64_t n = hs.size();
        f.write(reinterpret_cast<const char*>(&n), 8);
        f.write(hs.data(), (std::streamsize) hs.size());
        std::vector<uint16_t> b;
        for (const Tensor& t : ts) {
            b.resize(t.v.size());
            for (size_t i = 0; i < t.v.size(); ++i) b[i] = kernels::bf16_from_f32(t.v[i]);
            f.write(reinterpret_cast<const char*>(b.data()), (std::streamsize) (b.size() * 2));
        }
        if (!f) throw std::runtime_error("image engine: short write " + tmp);
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("image engine: cannot rename " + tmp);
}

}  // namespace strata::image::train
