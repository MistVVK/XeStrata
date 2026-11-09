// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_train_main.cpp - strata-image-qi-train: a LoRA on Qwen-Image-2.1, as ai-toolkit trains one.
//
//   strata-image-qi-train prep --te <Qwen3-VL GGUF> --vae <safetensors> --list <file>
//   strata-image-qi-train train --dit <F16 GGUF> --data <items file> --out <prefix> [--rank 32] [--steps 3000]
//                               [--lr 1e-4] [--seed 42] [--draw balanced|content|style] [--save-every 250]
//                               [--caption-dropout 0.05]
//   strata-image-qi-train check --dit <F16 GGUF> --dir <dir>
//
// prep is the caches ai-toolkit fills before it trains (tools/image/qi_train_prep.py writes the list and drives it):
// a line "img <rgb8 file> <W> <H> <out>" is an image already scaled and cropped to its bucket, encoded by the VAE to
// <out> (W / 16, H / 16, then the posterior's mean and standard deviation, normalized, [64][h][w] FP32 each); a line
// "txt <int32 ids file> <drop> <out>" is a caption's chat template's ids, its text encoder states past the first
// `drop` (the system turn) written to <out> (n, then [n][4096] FP32).
//
// train reads an items file, a line "<latent cache> <states cache>" each and a first line "empty <states cache>" (the
// caption " " the dropout swaps in), and runs ai-toolkit's loop: each epoch the items in a new order, a caption
// dropped with its probability, a posterior draw of the latent and new noise, the schedule's index drawn, one step
// and the clipped AdamW update; the LoRA saved every --save-every steps and at the end (<prefix>_<step>.safetensors).
//
// check is one step on the inputs tools/image/qi_train_step_ref.py writes (states, x0, noise, the LoRA factors), its
// loss, prediction and gradients written back for the reference to compare.
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/qi_text.hpp"
#include "strata/image/qi_trainer.hpp"
#include "strata/image/qi_vae.hpp"
#include "strata/image/train_loop.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace strata::image;
namespace gpu = strata::gpu;

void usage() {
    std::fprintf(stderr,
                 "usage: xestrata-image-qi-train prep --te <gguf> --vae <safetensors> --list <file>\n"
                 "       xestrata-image-qi-train train --dit <gguf> --data <items> --out <prefix> [--rank N] "
                 "[--steps N] [--lr F] [--seed N] [--draw balanced|content|style] [--save-every N] "
                 "[--caption-dropout F]\n"
                 "       xestrata-image-qi-train check --dit <gguf> --dir <dir> [--blocks N]\n");
}

std::map<std::string, std::string> flags(int argc, char** argv, int from) {
    std::map<std::string, std::string> f;
    for (int i = from; i < argc; ++i) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0 || i + 1 >= argc) throw std::invalid_argument("bad argument: " + k);
        f[k.substr(2)] = argv[++i];
    }
    return f;
}

std::string need(const std::map<std::string, std::string>& f, const std::string& k) {
    auto it = f.find(k);
    if (it == f.end()) throw std::invalid_argument("missing --" + k);
    return it->second;
}

std::string opt(const std::map<std::string, std::string>& f, const std::string& k, const std::string& d) {
    auto it = f.find(k);
    return it == f.end() ? d : it->second;
}

template<class T> std::vector<T> read_all(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<T> v(raw.size() / sizeof(T));
    std::memcpy(v.data(), raw.data(), v.size() * sizeof(T));
    return v;
}

template<class T> void write_all(const std::string& path, const T* p, size_t n) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(p), (std::streamsize) (n * sizeof(T)));
    if (!f) throw std::runtime_error("cannot write " + path);
}

// a latent cache: h, w, then the mean and the standard deviation [64][h][w]
struct Latent {
    int64_t h = 0, w = 0;
    std::vector<float> mean, sd;
};

Latent read_latent(const std::string& path) {
    std::vector<float> v = read_all<float>(path);
    if (v.size() < 2) throw std::runtime_error("short latent cache " + path);
    Latent l;
    l.w = (int64_t) v[0];
    l.h = (int64_t) v[1];
    const size_t n = (size_t) (64 * l.h * l.w);
    if (v.size() != 2 + 2 * n) throw std::runtime_error("latent cache of the wrong size " + path);
    l.mean.assign(v.begin() + 2, v.begin() + 2 + (long) n);
    l.sd.assign(v.begin() + 2 + (long) n, v.end());
    return l;
}

std::vector<float> read_states(const std::string& path, int64_t& n) {
    std::vector<float> v = read_all<float>(path);
    if (v.empty()) throw std::runtime_error("empty states cache " + path);
    n = (int64_t) v[0];
    if ((int64_t) v.size() != 1 + n * 4096) throw std::runtime_error("states cache of the wrong size " + path);
    return {v.begin() + 1, v.end()};
}

int prep(const std::map<std::string, std::string>& f, void* s) {
    std::ifstream list(need(f, "list"));
    if (!list) throw std::runtime_error("cannot read the list");
    std::vector<std::vector<std::string>> imgs, txts;
    for (std::string line; std::getline(list, line);) {
        std::istringstream ls(line);
        std::vector<std::string> w;
        for (std::string t; ls >> t;) w.push_back(t);
        if (w.empty()) continue;
        if (w[0] == "img" && w.size() == 5) imgs.push_back(w);
        else if (w[0] == "txt" && w.size() == 4) txts.push_back(w);
        else throw std::runtime_error("bad list line: " + line);
    }
    std::string err;
    if (!imgs.empty()) {   // the VAE first, then the text encoder: one of them on the GPU at a time
        QiVae vae;
        if (!vae.open(need(f, "vae"), s, err)) throw std::runtime_error(err);
        for (const auto& w : imgs) {
            const int64_t W = std::stoll(w[2]), H = std::stoll(w[3]), HW = W * H;
            std::vector<uint8_t> px = read_all<uint8_t>(w[1]);
            if ((int64_t) px.size() != HW * 3) throw std::runtime_error("bad image file " + w[1]);
            std::vector<float> rgba((size_t) (4 * HW), 1.0f);   // an opaque alpha, as ai-toolkit gives an RGB image
            for (int64_t q = 0; q < HW; ++q)
                for (int c = 0; c < 3; ++c) rgba[(size_t) (c * HW + q)] = (float) px[(size_t) (q * 3 + c)] / 255.f;
            const int64_t n = 64 * HW / 256;
            float *d = nullptr, *z = nullptr, *sd = nullptr;
            if (!gpu::alloc_device(&d, rgba.size() * 4) || !gpu::alloc_device(&z, (size_t) n * 4) ||
                !gpu::alloc_device(&sd, (size_t) n * 4))
                throw std::runtime_error("GPU memory for an image");
            gpu::copy(d, rgba.data(), rgba.size() * 4);
            if (!vae.encode(d, H, W, z, sd)) throw std::runtime_error("the VAE has no encoder");
            std::vector<float> out((size_t) (2 + 2 * n));
            out[0] = (float) (W / 16);
            out[1] = (float) (H / 16);
            gpu::copy(out.data() + 2, z, (size_t) n * 4);
            gpu::copy(out.data() + 2 + n, sd, (size_t) n * 4);
            gpu::free(d);
            gpu::free(z);
            gpu::free(sd);
            vae.drop_scratch();
            write_all(w[4], out.data(), out.size());
            std::printf("img %s %lldx%lld\n", w[4].c_str(), (long long) W, (long long) H);
        }
    }
    if (!txts.empty()) {
        QiText te;
        if (!te.open(need(f, "te"), s, err)) throw std::runtime_error(err);
        te.set_layer_budget(UINT64_MAX);
        for (const auto& w : txts) {
            const std::vector<int32_t> ids = read_all<int32_t>(w[1]);
            const int64_t drop = std::stoll(w[2]);
            std::vector<float> st = te.encode(ids);
            const int64_t D = te.hidden(), n = (int64_t) ids.size() - drop;
            if (D != 4096 || n <= 0) throw std::runtime_error("the text encoder's states do not fit " + w[1]);
            std::vector<float> out((size_t) (1 + n * D));
            out[0] = (float) n;
            std::copy(st.begin() + drop * D, st.end(), out.begin() + 1);
            write_all(w[3], out.data(), out.size());
            std::printf("txt %s %lld rows\n", w[3].c_str(), (long long) n);
        }
    }
    return 0;
}

int train_cmd(const std::map<std::string, std::string>& f, void* s) {
    const int64_t rank = std::stoll(opt(f, "rank", "32")), steps = std::stoll(opt(f, "steps", "3000"));
    const int64_t save_every = std::stoll(opt(f, "save-every", "250"));
    const uint64_t seed = std::stoull(opt(f, "seed", "42"));
    const float dropout = std::stof(opt(f, "caption-dropout", "0.05"));
    train::AdamW o;
    o.lr = std::stof(opt(f, "lr", "1e-4"));
    train::IndexDraw draw;
    if (!train::parse_index_draw(opt(f, "draw", "balanced"), draw)) throw std::invalid_argument("bad --draw");
    const std::string out = need(f, "out");
    QiTrainer tr;
    std::string err;
    if (!tr.open(need(f, "dit"), rank, seed, s, err)) throw std::runtime_error(err);
    std::ifstream items(need(f, "data"));
    if (!items) throw std::runtime_error("cannot read the items");
    struct Item {
        Latent lat;
        int text = -1;
    };
    std::vector<Item> all;
    int empty = -1;
    std::map<std::string, int> text_of;
    auto text = [&](const std::string& path) {
        auto it = text_of.find(path);
        if (it != text_of.end()) return it->second;
        int64_t n = 0;
        std::vector<float> st = read_states(path, n);
        return text_of[path] = tr.add_text(st, n);
    };
    for (std::string line; std::getline(items, line);) {
        std::istringstream ls(line);
        std::string a, b;
        if (!(ls >> a >> b)) continue;
        if (a == "empty") empty = text(b);
        else all.push_back({read_latent(a), text(b)});
    }
    if (all.empty() || empty < 0) throw std::runtime_error("the items file needs its \"empty\" line and an item");
    std::printf("qwen-image-2.1 lora: %zu items, rank %lld, %lld steps, lr %g, %s\n", all.size(), (long long) rank,
                (long long) steps, (double) o.lr, opt(f, "draw", "balanced").c_str());
    std::mt19937_64 rng(seed ^ 0x9e3779b97f4a7c15ull);   // the loop's own stream, not the LoRA's first values'
    std::normal_distribution<float> gauss(0.f, 1.f);
    std::uniform_real_distribution<float> uni(0.f, 1.f);
    std::vector<size_t> order;
    size_t next = 0;
    std::map<std::pair<int64_t, int64_t>, train::Schedule> sched;
    std::vector<float> x0, nz;
    const auto t0 = std::chrono::steady_clock::now();
    double loss_avg = 0.0;
    for (int64_t step = 1; step <= steps; ++step) {
        if (next == order.size()) {   // a new epoch: every item once, in a new order
            order.resize(all.size());
            for (size_t i = 0; i < order.size(); ++i) order[i] = i;
            std::shuffle(order.begin(), order.end(), rng);
            next = 0;
        }
        const Item& it = all[order[next++]];
        const int64_t h = it.lat.h, w = it.lat.w, hw = h * w;
        const int txt = uni(rng) < dropout ? empty : it.text;
        auto sk = sched.find({h, w});
        if (sk == sched.end()) sk = sched.emplace(std::make_pair(h, w), qi_train_schedule(h, w)).first;
        const float timestep = sk->second.t[(size_t) train::draw_index(rng, draw)];
        // the posterior's draw and the noise, as the latent's tokens [h w][64]
        x0.resize((size_t) (hw * 64));
        nz.resize(x0.size());
        for (int64_t c = 0; c < 64; ++c)
            for (int64_t p = 0; p < hw; ++p) {
                const size_t src = (size_t) (c * hw + p), dst = (size_t) (p * 64 + c);
                x0[dst] = it.lat.mean[src] + it.lat.sd[src] * gauss(rng);
            }
        for (float& v : nz) v = gauss(rng);
        const float loss = tr.step(txt, x0.data(), nz.data(), h, w, timestep);
        const double norm = tr.update(o);
        loss_avg = step == 1 ? loss : 0.98 * loss_avg + 0.02 * loss;
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("step %lld loss %.5f (avg %.5f) t %.1f grad %.3f %lldx%lld %.2f s/step\n", (long long) step,
                    (double) loss, loss_avg, (double) timestep, norm, (long long) (w * 16), (long long) (h * 16),
                    secs / (double) step);
        std::fflush(stdout);
        if (step % save_every == 0 || step == steps) {
            const std::string path = out + "_" + std::to_string(step) + ".safetensors";
            // ai-toolkit's metadata (get_meta_for_safetensors: the job's name and version, the software, format pt)
            // and what the run was
            const std::string name = out.substr(out.find_last_of('/') + 1);
            tr.save(path, {{"name", name}, {"version", "1.0"}, {"software", "{\"name\": \"XeStrata\"}"},
                           {"format", "pt"}, {"ss_network_module", "networks.lora"},
                           {"ss_network_dim", std::to_string(rank)}, {"ss_network_alpha", std::to_string(rank)},
                           {"ss_base_model", "qwen-image-2.1"}, {"ss_steps", std::to_string(step)},
                           {"ss_seed", std::to_string(seed)}, {"ss_learning_rate", opt(f, "lr", "1e-4")},
                           {"ss_content_or_style", opt(f, "draw", "balanced")}});
            std::printf("saved %s\n", path.c_str());
        }
    }
    return 0;
}

int check(const std::map<std::string, std::string>& f, void* s) {
    const std::string dir = need(f, "dir");
    const std::vector<float> p = read_all<float>(dir + "/params.bin");   // n, h, w, timestep, rank
    if (p.size() != 5) throw std::runtime_error("params.bin");
    const int64_t n = (int64_t) p[0], h = (int64_t) p[1], w = (int64_t) p[2], rank = (int64_t) p[4];
    QiTrainer tr;
    std::string err;
    if (!tr.open(need(f, "dit"), rank, 1, s, err, std::stoll(opt(f, "blocks", "0")))) throw std::runtime_error(err);
    const int text = tr.add_text(read_all<float>(dir + "/states.bin"), n);
    // the factors in grads()' order: per block, per product, A then B
    std::vector<train::Tensor> ts = tr.grads();
    const std::vector<float> fac = read_all<float>(dir + "/factors.bin");
    size_t off = 0;
    for (train::Tensor& t : ts) {
        if (off + t.v.size() > fac.size()) throw std::runtime_error("factors.bin is short");
        std::copy(fac.begin() + (long) off, fac.begin() + (long) (off + t.v.size()), t.v.begin());
        off += t.v.size();
    }
    if (off != fac.size()) throw std::runtime_error("factors.bin is long");
    tr.set_factors(ts);
    const std::vector<float> x0 = read_all<float>(dir + "/x0.bin"), nz = read_all<float>(dir + "/noise.bin");
    const auto t0 = std::chrono::steady_clock::now();
    const float loss = tr.step(text, x0.data(), nz.data(), h, w, p[3]);
    std::printf("loss %.6f in %.2f s\n", (double) loss,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    write_all(dir + "/eng_loss.bin", &loss, 1);
    const std::vector<float> pred = tr.last_pred();
    write_all(dir + "/eng_pred.bin", pred.data(), pred.size());
    std::vector<float> g;
    for (const train::Tensor& t : tr.grads()) g.insert(g.end(), t.v.begin(), t.v.end());
    write_all(dir + "/eng_grads.bin", g.data(), g.size());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 2;
    }
    try {
        const std::string op = argv[1];
        const auto f = flags(argc, argv, 2);
        void* s = strata::core::Runtime::get().create_stream();
        if (op == "prep") return prep(f, s);
        if (op == "train") return train_cmd(f, s);
        if (op == "check") return check(f, s);
        usage();
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata-image-qi-train: %s\n", e.what());
        return 1;
    }
}
