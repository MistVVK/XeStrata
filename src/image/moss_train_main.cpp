// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/moss_train_main.cpp - xestrata-image-moss-train: a LoRA on MOSS-Image as ai-toolkit's moss_image job
// trains one (include/strata/image/moss_trainer.hpp), from an export dir (tools/image/mf_pre.py --cache ms, then
// tools/image/mf_pre_export.py --cache ms).
//
// ai-toolkit's defaults for the model: the timesteps of its flow-match scheduler with a static shift of 5 (an index
// drawn uniform, randint(0, 999)), rank 32 with alpha = rank, AdamW lr 1e-4 (eps 1e-6, weight decay 1e-4) at a
// constant rate after torch's clip at 1, batch 1, 3,000 steps, a save every 250, the caption dropped 5% of the time,
// the base quantized (convrot8, int8 here), layers.22 left out (the block guidance's negative branch skips).
//
//   train --dit DIR --data EXPORT --out PREFIX [--rank 32] [--steps 3000] [--save-every 250] [--lr 1e-4]
//         [--seed 42] [--caption-dropout 0.05] [--base int8|bf16] [--skip 22] [--shift 5]
//   loss  --dit DIR --data EXPORT --stem STEM [--sigma 0.5] [--seed 42] [--base int8|bf16]
#include "strata/core/runtime.hpp"
#include "strata/image/moss_trainer.hpp"


#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace strata::image;

void usage() {
    std::fprintf(stderr,
                 "usage: xestrata-image-moss-train train --dit DIR --data EXPORT --out PREFIX [--rank N] [--steps N] "
                 "[--save-every N] [--lr F] [--seed N] [--caption-dropout F] [--base int8|bf16] [--skip LIST] "
                 "[--shift F]\n"
                 "       xestrata-image-moss-train loss --dit DIR --data EXPORT --stem STEM [--sigma F] [--seed N] "
                 "[--base int8|bf16]\n");
}

std::string opt(const std::map<std::string, std::string>& f, const std::string& k, const std::string& d) {
    auto it = f.find(k);
    return it == f.end() ? d : it->second;
}

std::string need(const std::map<std::string, std::string>& f, const std::string& k) {
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

// the step's latent drawn from the posterior where the export has its deviation
void draw_x0(const MfTrainSample& smp, std::mt19937_64& rng, std::vector<float>& x0) {
    std::normal_distribution<float> gauss(0.f, 1.f);
    x0 = smp.lat;
    if (smp.lat_sd.size() == smp.lat.size())
        for (size_t i = 0; i < x0.size(); ++i) x0[i] += smp.lat_sd[i] * gauss(rng);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 2;
    }
    const std::string cmd = argv[1];
    std::map<std::string, std::string> f;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k.rfind("--", 0) != 0) {
            usage();
            return 2;
        }
        f[k.substr(2)] = argv[i + 1];
    }
    try {
        void* s = strata::core::Runtime::get().create_stream();
        const uint64_t seed = std::stoull(opt(f, "seed", "42"));
        const std::string base = opt(f, "base", "int8");
        if (base != "int8" && base != "bf16") throw std::invalid_argument("--base is int8 or bf16");
        const std::string data = need(f, "data");
        MossTrainer tr;
        std::string err;
        const bool train = cmd == "train";
        if (!train && cmd != "loss") {
            usage();
            return 2;
        }
        const int64_t rank = train ? std::stoll(opt(f, "rank", "32")) : 0;
        if (!tr.open(need(f, "dit"), rank, base == "int8", int_list(opt(f, "skip", "22")), seed, s, err))
            throw std::runtime_error(err);
        std::mt19937_64 rng(seed ^ 0x9e3779b97f4a7c15ull);   // the loop's own stream, not the LoRA's first values'
        std::normal_distribution<float> gauss(0.f, 1.f);
        std::vector<float> x0, eps;
        if (!train) {
            const MfTrainSample smp = mf_trainer_sample(data, need(f, "stem"), tr.hidden());
            const float sigma = std::stof(opt(f, "sigma", "0.5"));
            x0 = smp.lat;
            eps.resize(x0.size());
            for (float& v : eps) v = gauss(rng);
            std::printf("loss: %.6f (sigma %.3f, %lld tokens, %lld text rows)\n",
                        (double) tr.step(smp, x0, eps, sigma, false), (double) sigma, (long long) smp.tokens,
                        (long long) smp.rows);
            return 0;
        }
        const int64_t steps = std::stoll(opt(f, "steps", "3000")), save_every = std::stoll(opt(f, "save-every", "250"));
        const float dropout = std::stof(opt(f, "caption-dropout", "0.05"));
        const double shift = std::stod(opt(f, "shift", "5"));
        strata::image::train::AdamW o;
        o.lr = std::stof(opt(f, "lr", "1e-4"));
        const std::string out = need(f, "out");
        const std::vector<std::string> stems = stems_of(data);
        const strata::image::train::Schedule sched = strata::image::train::schedule_shift(shift);
        std::printf("moss-image lora: %zu samples, rank %lld, %lld steps, lr %g, shift %g, base %s, skip %s\n",
                    stems.size(), (long long) rank, (long long) steps, (double) o.lr, shift, base.c_str(),
                    opt(f, "skip", "22").c_str());
        std::uniform_real_distribution<float> uni(0.f, 1.f);
        std::vector<size_t> order;
        size_t next = 0;
        const auto t0 = std::chrono::steady_clock::now();
        double loss_avg = 0.0;
        for (int64_t step = 1; step <= steps; ++step) {
            if (next == order.size()) {   // a new epoch: every sample once, in a new order
                order.resize(stems.size());
                for (size_t i = 0; i < order.size(); ++i) order[i] = i;
                std::shuffle(order.begin(), order.end(), rng);
                next = 0;
            }
            MfTrainSample smp = mf_trainer_sample(data, stems[order[next++]], tr.hidden());
            if (uni(rng) < dropout) {   // the caption dropped: the empty prompt's rows
                smp.te = smp.uncond;
                smp.rows = smp.uncond_rows;
            }
            const float sigma =
                sched.t[(size_t) strata::image::train::draw_index(rng, strata::image::train::IndexDraw::balanced)] /
                1000.f;
            draw_x0(smp, rng, x0);
            eps.resize(x0.size());
            for (float& v : eps) v = gauss(rng);
            const auto ts = std::chrono::steady_clock::now();
            const float loss = tr.step(smp, x0, eps, sigma, true);
            const double norm = tr.adamw(o);
            loss_avg = step == 1 ? loss : 0.98 * loss_avg + 0.02 * loss;
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const double own = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
            std::printf("step %lld loss %.5f (avg %.5f) sigma %.3f grad %.3f %lldx%lld %.2f s (%.2f s/step)\n",
                        (long long) step, (double) loss, loss_avg, (double) sigma, norm, (long long) (smp.w * 16),
                        (long long) (smp.h * 16), own, secs / (double) step);
            std::fflush(stdout);
            if ((save_every > 0 && step % save_every == 0) || step == steps) {
                const std::string path = out + "_" + std::to_string(step) + ".safetensors";
                const std::string name = out.substr(out.find_last_of('/') + 1);
                tr.save(path, {{"name", name}, {"version", "1.0"}, {"software", "{\"name\": \"XeStrata\"}"},
                               {"format", "pt"}, {"ss_base_model_version", "moss_image"},
                               {"ss_network_module", "networks.lora"}, {"ss_network_dim", std::to_string(rank)},
                               {"ss_network_alpha", std::to_string(rank)}, {"ss_steps", std::to_string(step)},
                               {"ss_seed", std::to_string(seed)}, {"ss_learning_rate", opt(f, "lr", "1e-4")}});
                std::printf("saved %s\n", path.c_str());
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "moss-train: %s\n", e.what());
        return 1;
    }
    return 0;
}
