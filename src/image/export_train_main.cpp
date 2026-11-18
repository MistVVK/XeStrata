// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/export_train_main.cpp - xestrata-image-moss-train and xestrata-image-anima-train: a LoRA on MOSS-Image
// (include/strata/image/moss_trainer.hpp) or Anima (include/strata/image/anima_trainer.hpp) as ai-toolkit's job for
// the model trains one, from an export dir (tools/image/mf_pre.py --cache ms or an, then tools/image/mf_pre_export.py).
// The executable's model is STRATA_TRAIN_MODEL ("moss-image" or "anima"), set by CMake.
//
// ai-toolkit's defaults for both: rank 32 with alpha = rank, AdamW lr 1e-4 (eps 1e-6, weight decay 1e-4) at a constant
// rate after torch's clip at 1, batch 1, 3,000 steps, a save every 250, and an index
// drawn uniform, randint(0, 999), into the model's timesteps.  The models' own:
//   MOSS-Image: its flow-match scheduler's timesteps with a static shift of 5, the base quantized (convrot8, int8
//               here), layers.22 left out (the block guidance's negative branch skips);
//   Anima:      its "weighted" timesteps (the 1,000 linear ones, sigma = t / 1000 with no shift, the loss times
//               ai-toolkit's weight of the index), the base unquantized.
//
//   train --dit DIT --data EXPORT --out PREFIX [--rank 32] [--steps 3000] [--save-every 250] [--lr 1e-4] [--seed 42]
//         [--caption-dropout 0] [--train-blocks 0-5,10] [--only-if-contains PARTS] [--ignore-if-contains PARTS]
//         [--kind character|style|other] [--trigger PHRASE] [--vram-budget MIB] [--vram-control FILE] [--state FILE] [--resume FILE]
//         MOSS-Image also [--base int8|bf16] [--skip 22] [--shift 5]
//   (the weights trained: train::Targets, every one by default; --skip the blocks left out besides; --kind what the
//   LoRA adds, in its metadata)
//   loss  --dit DIT --data EXPORT --stem STEM [--sigma 0.5] [--seed 42]   MOSS-Image also [--base int8|bf16]
#include "strata/core/runtime.hpp"
#include "strata/image/anima_trainer.hpp"
#include "strata/image/moss_trainer.hpp"
#include "strata/image/train_control.hpp"
#include "strata/image/train_loop.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef STRATA_TRAIN_MODEL
#error "STRATA_TRAIN_MODEL names the executable's model"
#endif

namespace {

using namespace strata::image;
using Flags = std::map<std::string, std::string>;

constexpr bool kMoss = std::string_view(STRATA_TRAIN_MODEL) == "moss-image";

void usage() {
    if (kMoss)
        std::fprintf(stderr, "usage: xestrata-image-moss-train train --dit DIR --data EXPORT --out PREFIX [--rank N] "
                             "[--steps N] [--save-every N] [--lr F] [--seed N] [--caption-dropout F] "
                             "[--train-blocks LIST] [--only-if-contains PARTS] [--ignore-if-contains PARTS] "
                             "[--kind character|style|other] [--trigger PHRASE] [--vram-budget MIB] [--vram-control FILE] [--state FILE] "
                             "[--resume FILE] [--base int8|bf16] [--skip LIST] [--shift F]\n"
                             "       xestrata-image-moss-train loss --dit DIR --data EXPORT --stem STEM [--sigma F] "
                             "[--seed N] [--base int8|bf16]\n");
    else
        std::fprintf(stderr, "usage: xestrata-image-anima-train train --dit FILE --data EXPORT --out PREFIX [--rank N] "
                             "[--steps N] [--save-every N] [--lr F] [--seed N] [--caption-dropout F] "
                             "[--train-blocks LIST] [--only-if-contains PARTS] [--ignore-if-contains PARTS] "
                             "[--kind character|style|other] [--trigger PHRASE] [--vram-budget MIB] [--vram-control FILE] [--state FILE] "
                             "[--resume FILE]\n"
                             "       xestrata-image-anima-train loss --dit FILE --data EXPORT --stem STEM [--sigma F] "
                             "[--seed N]\n");
}

std::string opt(const Flags& f, const std::string& k, const std::string& d) {
    auto it = f.find(k);
    return it == f.end() ? d : it->second;
}

std::string need(const Flags& f, const std::string& k) {
    auto it = f.find(k);
    if (it == f.end()) throw std::invalid_argument("--" + k + " is needed");
    return it->second;
}

// the manifest's stems (its comment lines skipped)
std::vector<std::string> stems_of(const std::string& dir) {
    std::ifstream mf(dir + "/manifest.tsv");
    if (!mf) throw std::runtime_error(dir + ": no manifest.tsv");
    std::vector<std::string> out;
    for (std::string line; std::getline(mf, line);) {
        if (line.empty() || line[0] == '#') continue;
        out.push_back(line.substr(0, line.find('\t')));
    }
    if (out.empty()) throw std::runtime_error(dir + ": no samples");
    return out;
}

std::vector<int> int_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, ',');)
        if (!t.empty()) out.push_back(std::stoi(t));
    return out;
}

// The loop, the same for both models: each epoch the samples in a new order, a caption dropped with its probability,
// the schedule's index drawn, a posterior draw of the latent and new noise, one step and the clipped AdamW update; the
// LoRA saved every --save-every steps and at the end (<prefix>_<step>.safetensors).
template <class T>
int run(T& tr, const Flags& f, bool train, uint64_t seed, int64_t rank, const train::Targets& targets) {
    const std::string data = need(f, "data");
    if (const std::string b = opt(f, "vram-budget", "0"); b != "0")
        tr.base().set_vram_budget((uint64_t) std::stoll(b) << 20);   // the test's fixed form (loss included)
    std::mt19937_64 rng(seed ^ 0x9e3779b97f4a7c15ull);   // the loop's own stream, not the LoRA's first values'
    std::normal_distribution<float> gauss(0.f, 1.f);
    std::vector<float> x0, eps;
    if (!train) {
        const MfTrainSample smp = mf_trainer_sample(data, need(f, "stem"), tr.text_cols());
        const float sigma = std::stof(opt(f, "sigma", "0.5"));
        x0 = smp.lat;
        eps.resize(x0.size());
        for (float& v : eps) v = gauss(rng);
        std::printf("loss: %.6f (sigma %.3f, %lld tokens, %lld text rows)\n",
                    (double) tr.step(smp, x0, eps, sigma, 1.f, false), (double) sigma, (long long) smp.tokens,
                    (long long) smp.rows);
        return 0;
    }
    const int64_t steps = std::stoll(opt(f, "steps", "3000")), save_every = std::stoll(opt(f, "save-every", "250"));
    const std::string dropout_arg = opt(f, "caption-dropout", "");
    const float dropout = dropout_arg.empty() ? train::caption_dropout : std::stof(dropout_arg);
    train::AdamW o;
    o.lr = std::stof(opt(f, "lr", "1e-4"));
    const std::string out = need(f, "out");
    const std::vector<std::string> stems = stems_of(data);
    const double shift = std::stod(opt(f, "shift", "5"));
    const train::Schedule sched = kMoss ? train::schedule_shift(shift) : train::schedule_linear();
    if (kMoss)
        std::printf("moss-image lora: %zu samples, rank %lld, %lld steps, lr %g, shift %g, base %s, %s\n",
                    stems.size(), (long long) rank, (long long) steps, (double) o.lr, shift,
                    opt(f, "base", "int8").c_str(), targets.text().c_str());
    else
        std::printf("anima lora: %zu samples, rank %lld, %lld steps, lr %g, weighted timesteps%s%s\n", stems.size(),
                    (long long) rank, (long long) steps, (double) o.lr, targets.text().empty() ? "" : ", ",
                    targets.text().c_str());
    std::uniform_real_distribution<float> uni(0.f, 1.f);
    std::vector<size_t> order;
    size_t next = 0;
    double loss_avg = 0.0;
    int64_t done = 0;   // the steps completed (the state file's; --resume continues after them)
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
    co.initial = train::initial_vram(kMoss ? "moss-image" : "anima", kMoss ? opt(f, "base", "int8") : "bf16",
                                                mf_train_max_tokens(data), rank);
    train::Control ctl(tr.base(), co);
    ctl.load(opt(f, "resume", ""), load_loop);
    train::log_training_plan(tr.base(), o.lr);
    ctl.report();
    const int64_t start_step = done + 1;
    const auto t0 = std::chrono::steady_clock::now();
    for (int64_t step = start_step; step <= steps; ++step) {
        done = step - 1;
        if (!ctl.boundary(true, save_loop)) return 0;   // the budget, pause/resume, the state file (every step ends in an update)
        if (next == order.size()) {   // a new epoch: every sample once, in a new order
            order.resize(stems.size());
            for (size_t i = 0; i < order.size(); ++i) order[i] = i;
            std::shuffle(order.begin(), order.end(), rng);
            next = 0;
        }
        MfTrainSample smp = mf_trainer_sample(data, stems[order[next++]], tr.text_cols());
        if (uni(rng) < dropout) {   // the caption dropped: the empty prompt's rows
            smp.te = smp.uncond;
            smp.rows = smp.uncond_rows;
        }
        const int64_t idx = train::draw_index(rng, train::IndexDraw::balanced);
        const float sigma = sched.t[(size_t) idx] / 1000.f;
        x0 = smp.lat;   // the step's latent drawn from the posterior where the export has its deviation
        if (smp.lat_sd.size() == x0.size())
            for (size_t i = 0; i < x0.size(); ++i) x0[i] += smp.lat_sd[i] * gauss(rng);
        eps.resize(x0.size());
        for (float& v : eps) v = gauss(rng);
        const auto ts = std::chrono::steady_clock::now();
        const float loss = tr.step(smp, x0, eps, sigma, kMoss ? 1.f : train::flow_weight(idx), true);
        if (!std::isfinite(loss)) throw std::runtime_error("training: non-finite loss");
        const double norm = tr.adamw(o);
        loss_avg = step == 1 ? loss : 0.98 * loss_avg + 0.02 * loss;
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double own = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
        std::printf("step %lld loss %.5f (avg %.5f) sigma %.3f grad %.3f %lldx%lld %.2f s (%.2f s/step)\n",
                    (long long) step, (double) loss, loss_avg, (double) sigma, norm, (long long) smp.w * 16,
                    (long long) smp.h * 16, own, secs / (double) (step - start_step + 1));
        std::fflush(stdout);
        ctl.report();
        if ((save_every > 0 && step % save_every == 0) || step == steps) {
            const std::string path = out + "_" + std::to_string(step) + ".safetensors";
            train::LoraMeta meta;
            meta.name = out.substr(out.find_last_of('/') + 1);
            meta.base = kMoss ? "moss_image" : "anima";
            meta.lr = opt(f, "lr", "1e-4");
            meta.rank = rank;
            meta.alpha = (double) rank;
            meta.steps = step;
            meta.max_steps = steps;
            meta.seed = seed;
            meta.kind = opt(f, "kind", "");
            meta.trigger = opt(f, "trigger", "");
            meta.targets = targets.text();
            tr.save(path, train::lora_meta(meta));
            std::printf("saved %s\n", path.c_str());
        }
        ctl.progress(step, steps, loss);
        done = step;
    }
    ctl.save(save_loop);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    strata::image::train::install_stop_signals();   // stop at a boundary and unwind the GPU owners
    if (argc < 2) {
        usage();
        return 2;
    }
    const std::string cmd = argv[1];
    Flags f;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k.rfind("--", 0) != 0) {
            usage();
            return 2;
        }
        f[k.substr(2)] = argv[i + 1];
    }
    const bool train = cmd == "train";
    if (!train && cmd != "loss") {
        usage();
        return 2;
    }
    try {
        void* s = strata::core::Runtime::get().create_stream();
        const uint64_t seed = std::stoull(opt(f, "seed", "42"));
        const int64_t rank = train ? std::stoll(opt(f, "rank", "32")) : 0;
        std::string err;
        namespace tl = strata::image::train;   // `train` is the command's flag here
        tl::Targets targets;
        if (!tl::parse_targets(opt(f, "train-blocks", ""), opt(f, "only-if-contains", ""),
                               opt(f, "ignore-if-contains", ""), targets, err))
            throw std::invalid_argument(err);
        if (!tl::lora_kind_ok(opt(f, "kind", ""))) throw std::invalid_argument("--kind is character, style or other");
        if (kMoss) {
            const std::string base = opt(f, "base", "int8");
            if (base != "int8" && base != "bf16") throw std::invalid_argument("--base is int8 or bf16");
            MossTrainer tr;
            targets.skip = int_list(opt(f, "skip", "22"));
            if (!tr.open(need(f, "dit"), rank, base == "int8", targets, seed, s, err)) throw std::runtime_error(err);
            return run(tr, f, train, seed, rank, targets);
        }
        AnimaTrainer tr;
        if (!tr.open(need(f, "dit"), rank, seed, s, err, targets)) throw std::runtime_error(err);
        return run(tr, f, train, seed, rank, targets);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: %s\n", kMoss ? "moss-train" : "anima-train", e.what());
        return 1;
    }
}
