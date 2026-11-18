// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_train_main.cpp - strata-image-qi-train: a LoRA on Qwen-Image-2.1, as ai-toolkit trains one.
//
//   strata-image-qi-train train --dit <F16 GGUF> --data <items file> --out <prefix> [--rank 32] [--steps 3000]
//                               [--lr 1e-4] [--seed 42] [--draw balanced|content|style] [--save-every 250]
//                               [--caption-dropout 0] [--base bf16|int8] [--train-blocks 0-5,10]
//                               [--only-if-contains attn] [--ignore-if-contains img_mlp] [--kind character|style|other]
//   strata-image-qi-train check --dit <F16 GGUF> --dir <dir> [--blocks N] [--base bf16|int8]
//
// The caches the trainer reads are ai-toolkit's, made before it trains by tools/image/mf_pre.py --cache qi (the list's
// images through the VAE, its prompts through the text encoder; the reader is train_sample.hpp).
//
// train reads a tab-separated items file, a line "<latent cache>\t<states cache>" each and a first line
// "empty\t<states cache>" (the caption " " the dropout swaps in), and runs ai-toolkit's loop: each epoch the items in
// a new order, a caption
// dropped with its probability, a posterior draw of the latent and new noise, the schedule's index drawn, one step
// and the clipped AdamW update; the LoRA saved every --save-every steps and at the end (<prefix>_<step>.safetensors).
//
// --base int8 is a QLoRA: the blocks' products on ai-toolkit's convrot8 base (W8A8, made from the 16-bit file), as
// ai-toolkit's default trains.
//
// --train-blocks, --only-if-contains and --ignore-if-contains choose the products trained (train::Targets: every one
// by default); --kind says what the LoRA adds, in its metadata (ss_xestrata_kind), for serve's listing.
//
// check is one step on the inputs tools/image/qi_train_step_ref.py writes (states, x0, noise, the LoRA factors), its
// loss, prediction and gradients written back for the reference to compare.
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/qi_trainer.hpp"
#include "strata/image/train_sample.hpp"
#include "strata/image/train_loop.hpp"
#include "strata/image/train_control.hpp"

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
                 "usage: xestrata-image-qi-train train --dit <gguf> --data <items> --out <prefix> [--rank N] "
                 "[--steps N] [--lr F] [--seed N] [--draw balanced|content|style] [--save-every N] "
                 "[--caption-dropout F] [--base bf16|int8] [--train-blocks LIST] [--only-if-contains PARTS] "
                 "[--ignore-if-contains PARTS] [--kind character|style|other] [--trigger PHRASE] [--vram-budget MIB] "
                 "[--vram-control FILE] [--state FILE] [--resume FILE]\n"
                 "       xestrata-image-qi-train check --dit <gguf> --dir <dir> [--blocks N] [--base bf16|int8]\n");
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

int train_cmd(const std::map<std::string, std::string>& f, void* s) {
    const int64_t rank = std::stoll(opt(f, "rank", "32")), steps = std::stoll(opt(f, "steps", "3000"));
    const int64_t save_every = std::stoll(opt(f, "save-every", "250"));
    const uint64_t seed = std::stoull(opt(f, "seed", "42"));
    const std::string dropout_arg = opt(f, "caption-dropout", "");
    const float dropout = dropout_arg.empty() ? train::caption_dropout : std::stof(dropout_arg);
    train::AdamW o;
    o.lr = std::stof(opt(f, "lr", "1e-4"));
    train::IndexDraw draw;
    if (!train::parse_index_draw(opt(f, "draw", "balanced"), draw)) throw std::invalid_argument("bad --draw");
    const std::string out = need(f, "out");
    const std::string base = opt(f, "base", "bf16");
    if (base != "bf16" && base != "int8") throw std::invalid_argument("bad --base (bf16 or int8)");
    QiTrainer tr;
    std::string err;
    train::Targets targets;
    if (!train::parse_targets(opt(f, "train-blocks", ""), opt(f, "only-if-contains", ""),
                              opt(f, "ignore-if-contains", ""), targets, err))
        throw std::invalid_argument(err);
    const std::string kind = opt(f, "kind", "");
    if (!train::lora_kind_ok(kind)) throw std::invalid_argument("bad --kind (character, style or other)");
    if (!tr.open(need(f, "dit"), rank, seed, s, err, 0, base == "int8", targets)) throw std::runtime_error(err);
    std::ifstream items(need(f, "data"));
    if (!items) throw std::runtime_error("cannot read the items");
    struct Item {
        QiTrainLatent lat;
        int text = -1;
    };
    std::vector<Item> all;
    int empty = -1;
    int64_t max_text_rows = 0;
    std::map<std::string, int> text_of;
    auto text = [&](const std::string& path) {
        auto it = text_of.find(path);
        if (it != text_of.end()) return it->second;
        int64_t n = 0;
        std::vector<float> st = qi_train_states(path, n);
        max_text_rows = std::max(max_text_rows, n);
        return text_of[path] = tr.add_text(st, n);
    };
    for (std::string line; std::getline(items, line);) {
        // tab-separated: a cache path can hold spaces, never a tab.  A file written on Windows has CRLF lines.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream ls(line);
        std::string a, b;
        if (!std::getline(ls, a, '\t') || !std::getline(ls, b, '\t')) continue;
        if (a == "empty") empty = text(b);
        else all.push_back({qi_train_latent(a), text(b)});
    }
    if (all.empty() || empty < 0) throw std::runtime_error("the items file needs its \"empty\" line and an item");
    int64_t max_hw = 0;
    for (const Item& it : all) max_hw = std::max(max_hw, it.lat.h * it.lat.w);
    tr.plan_rows(max_hw + max_text_rows);
    std::printf("qwen-image-2.1 lora: %zu items, rank %lld, %lld steps, lr %g, %s\n", all.size(), (long long) rank,
                (long long) steps, (double) o.lr, opt(f, "draw", "balanced").c_str());
    std::mt19937_64 rng(seed ^ 0x9e3779b97f4a7c15ull);   // the loop's own stream, not the LoRA's first values'
    std::normal_distribution<float> gauss(0.f, 1.f);
    std::uniform_real_distribution<float> uni(0.f, 1.f);
    std::vector<size_t> order;
    size_t next = 0;
    double loss_avg = 0.0;
    int64_t done = 0;   // the steps completed (the state file's; --resume continues after them)
    // the loop's state as the state file holds it (the rng's whole state, the data's order and place)
    auto save_loop = [&](std::ostream& os) {
        os << done << '\n' << rng << '\n' << gauss << '\n' << order.size() << '\n';
        for (size_t v : order) os << v << ' ';
        os << '\n' << next << '\n' << loss_avg << '\n';
    };
    auto load_loop = [&](std::istream& is) {
        size_t on = 0;
        is >> done >> rng >> gauss >> on;
        order.resize(on);
        for (size_t& v : order) is >> v;
        is >> next >> loss_avg;
        if (!is) throw std::runtime_error("the state file's loop state did not read");
    };
    train::Control::Opts co;
    co.path = opt(f, "vram-control", "");
    co.budget_mib = (uint64_t) std::stoll(opt(f, "vram-budget", "0"));
    co.state = opt(f, "state", out + ".state.bin");
    co.initial = train::initial_vram("qwen-image-2.1", base, max_hw + max_text_rows, rank);
    train::Control ctl(tr.base(), co);
    ctl.load(opt(f, "resume", ""), load_loop);
    train::log_training_plan(tr.base(), o.lr);
    ctl.report();
    const int64_t start_step = done + 1;
    std::map<std::pair<int64_t, int64_t>, train::Schedule> sched;
    std::vector<float> x0, nz;
    const auto t0 = std::chrono::steady_clock::now();
    for (int64_t step = start_step; step <= steps; ++step) {
        done = step - 1;
        if (!ctl.boundary(true, save_loop)) return 0;   // the budget, pause/resume, the state file (every step ends in an update)
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
        const auto ts = std::chrono::steady_clock::now();
        const float loss = tr.step(txt, x0.data(), nz.data(), h, w, timestep);
        const double norm = tr.update(o);
        loss_avg = step == 1 ? loss : 0.98 * loss_avg + 0.02 * loss;
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double own = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
        std::printf("step %lld loss %.5f (avg %.5f) t %.1f grad %.3f %lldx%lld %.2f s (%.2f s/step)\n",
                    (long long) step, (double) loss, loss_avg, (double) timestep, norm, (long long) w * 16,
                    (long long) h * 16, own, secs / (double) (step - start_step + 1));
        std::fflush(stdout);
        ctl.report();
        if ((save_every > 0 && step % save_every == 0) || step == steps) {
            const std::string path = out + "_" + std::to_string(step) + ".safetensors";
            train::LoraMeta meta;
            meta.name = out.substr(out.find_last_of('/') + 1);
            meta.base = "qwen-image-2.1";
            meta.lr = opt(f, "lr", "1e-4");
            meta.rank = rank;
            meta.alpha = (double) rank;
            meta.steps = step;
            meta.max_steps = steps;
            meta.seed = seed;
            meta.kind = kind;
            meta.trigger = opt(f, "trigger", "");
            meta.targets = targets.text();
            meta.extra = {{"ss_content_or_style", opt(f, "draw", "balanced")}};
            tr.save(path, train::lora_meta(meta));
            std::printf("saved %s\n", path.c_str());
        }
        ctl.progress(step, steps, loss);
        done = step;
    }
    ctl.save(save_loop);
    return 0;
}

int check(const std::map<std::string, std::string>& f, void* s) {
    const std::string dir = need(f, "dir");
    const std::vector<float> p = read_all<float>(dir + "/params.bin");   // n, h, w, timestep, rank
    if (p.size() != 5) throw std::runtime_error("params.bin");
    const int64_t n = (int64_t) p[0], h = (int64_t) p[1], w = (int64_t) p[2], rank = (int64_t) p[4];
    QiTrainer tr;
    std::string err;
    if (!tr.open(need(f, "dit"), rank, 1, s, err, std::stoll(opt(f, "blocks", "0")), opt(f, "base", "bf16") == "int8"))
        throw std::runtime_error(err);
    if (const std::string b = opt(f, "vram-budget", "0"); b != "0")
        tr.base().set_vram_budget((uint64_t) std::stoll(b) << 20);   // the test's fixed form (byte-match)
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
    strata::image::train::install_stop_signals();   // stop at a boundary and unwind the GPU owners
    if (argc < 2) {
        usage();
        return 2;
    }
    try {
        const std::string op = argv[1];
        const auto f = flags(argc, argv, 2);
        void* s = strata::core::Runtime::get().create_stream();
        if (op == "train") return train_cmd(f, s);
        if (op == "check") return check(f, s);
        usage();
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "xestrata-image-qi-train: %s\n", e.what());
        return 1;
    }
}
