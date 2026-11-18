// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/train_loop.cpp - see include/strata/image/train_loop.hpp.
#include "sigmas.hpp"
#include "strata/image/train_loop.hpp"

#include "strata/core/gpu.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/platform/replace.hpp"

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
    for (double& v : s) v = sigma_shift(v, shift);
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
    if (!strata::platform::replace_file(tmp, path)) throw std::runtime_error("image engine: cannot rename " + tmp);
}

std::vector<std::pair<std::string, std::string>> lora_meta(const LoraMeta& m) {
    std::ostringstream alpha;
    alpha << m.alpha;   // 32, or 16.5
    std::vector<std::pair<std::string, std::string>> out = {
        {"name", m.name},
        {"version", "1.0"},
        {"software", "{\"name\": \"XeStrata\"}"},
        {"format", "pt"},
        {"modelspec.title", m.name},
        {"ss_model", m.name},
        {"ss_network_module", m.module},
        {"ss_network_dim", std::to_string(m.rank)},
        {"ss_network_alpha", alpha.str()},
        {"ss_base_model_version", m.base},
        {"ss_steps", std::to_string(m.steps)},
        {"ss_max_steps", std::to_string(m.max_steps)},
        {"ss_seed", std::to_string(m.seed)},
        {"ss_learning_rate", m.lr},
        {"ss_optimizer_type", "AdamW"},
        {"ss_xestrata_revision", STRATA_VERSION},
    };
    if (!m.trigger.empty()) out.emplace_back("modelspec.trigger_phrase", m.trigger);
    if (!m.kind.empty()) out.emplace_back("ss_xestrata_kind", m.kind);
    if (!m.targets.empty()) out.emplace_back("ss_xestrata_targets", m.targets);
    out.insert(out.end(), m.extra.begin(), m.extra.end());
    return out;
}

namespace {

// a value's parts between commas, each trimmed, the empty ones dropped
std::vector<std::string> comma_parts(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (size_t i = 0; i <= s.size(); ++i) {
        if (i == s.size() || s[i] == ',') {
            const size_t a = cur.find_first_not_of(" \t"), b = cur.find_last_not_of(" \t");
            if (a != std::string::npos) out.push_back(cur.substr(a, b - a + 1));
            cur.clear();
        } else {
            cur += s[i];
        }
    }
    return out;
}

// a block's number: digits only
bool block_number(const std::string& s, int& out) {
    if (s.empty() || s.size() > 6 || s.find_first_not_of("0123456789") != std::string::npos) return false;
    out = std::stoi(s);
    return true;
}

}  // namespace

bool Targets::take(int block, const std::string& name) const {
    if (block >= 0) {
        if (std::find(skip.begin(), skip.end(), block) != skip.end()) return false;
        if (!blocks.empty() && std::none_of(blocks.begin(), blocks.end(), [&](const std::pair<int, int>& r) {
                return block >= r.first && block <= r.second;
            }))
            return false;
    }
    const auto in = [&](const std::string& part) { return name.find(part) != std::string::npos; };
    if (!only.empty() && std::none_of(only.begin(), only.end(), in)) return false;
    return std::none_of(ignore.begin(), ignore.end(), in);
}

std::string Targets::text() const {
    std::string out;
    const auto add = [&](const char* what, const std::string& v) {
        if (v.empty()) return;
        if (!out.empty()) out += "; ";
        out += std::string(what) + " " + v;
    };
    std::string b;
    for (const auto& r : blocks)
        b += (b.empty() ? "" : ",") + std::to_string(r.first) +
             (r.second != r.first ? "-" + std::to_string(r.second) : std::string());
    add("blocks", b);
    std::string k;
    for (int x : skip) k += (k.empty() ? "" : ",") + std::to_string(x);
    add("skip", k);
    std::string o, g;
    for (const std::string& x : only) o += (o.empty() ? "" : ",") + x;
    for (const std::string& x : ignore) g += (g.empty() ? "" : ",") + x;
    add("only_if_contains", o);
    add("ignore_if_contains", g);
    return out;
}

bool parse_targets(const std::string& blocks, const std::string& only, const std::string& ignore, Targets& out,
                   std::string& err) {
    out.blocks.clear();
    if (!blocks.empty() && blocks != "all") {
        for (const std::string& part : comma_parts(blocks)) {
            const size_t dash = part.find('-');
            int a = 0, b = 0;
            if (dash == std::string::npos ? !block_number(part, a)
                                          : !block_number(part.substr(0, dash), a) ||
                                                !block_number(part.substr(dash + 1), b)) {
                err = "--train-blocks: \"all\", or blocks and ranges with commas (0-5,10), not " + part;
                return false;
            }
            if (dash == std::string::npos) b = a;
            if (b < a) {
                err = "--train-blocks: a range from its first block to its last, not " + part;
                return false;
            }
            out.blocks.emplace_back(a, b);
        }
        if (out.blocks.empty()) {
            err = "--train-blocks: no block";
            return false;
        }
    }
    out.only = comma_parts(only);
    out.ignore = comma_parts(ignore);
    return true;
}

}  // namespace strata::image::train
