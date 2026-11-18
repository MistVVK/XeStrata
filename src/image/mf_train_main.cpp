// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/mf_train_main.cpp - strata-image-mf-train: Mage-Flow's smallest training loop (card 5).
//
//   strata-image-mf-train info --dit <safetensors>
//   strata-image-mf-train loss --dit <safetensors> --data <export dir> --stem <sample> [--sigma F] [--seed N]
//
// info names the model's widths; loss runs one forward (the loss of one noised sample, no training yet).
#include "strata/core/gpu.hpp"
#include "strata/core/runtime.hpp"
#include "strata/image/mf_trainer.hpp"
#include "strata/image/train_loop.hpp"
#include "strata/image/train_control.hpp"
#include "strata/prefill/gemm.hpp"

#include <algorithm>
#include <cstdio>
#include <istream>
#include <ostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

void usage() {
    std::fprintf(stderr,
                 "usage: xestrata-image-mf-train info --dit <safetensors>\n"
                 "       xestrata-image-mf-train loss --dit <safetensors> --data <export dir> --stem <sample> "
                 "[--sigma F] [--seed N]\n"
                 "       xestrata-image-mf-train train --dit <safetensors> --data <export dir> --out <prefix> "
                 "[--method lora|lokr|dora] [--rank N] [--alpha F] [--steps N] [--lr F] [--seed N] [--accum N] "
                 "[--checkpoint N] [--trigger S] [--title S] [--timesteps linear|logit-normal|shifted] [--sigma-max F] "
                 "[--caption-dropout F] [--warmup N] [--eps F] [--weight-decay F] [--mod 1|0] [--base int8|bf16] "
                 "[--train-blocks LIST] [--only-if-contains PARTS] [--ignore-if-contains PARTS] "
                 "[--kind character|style|other] [--vram-budget MIB] [--vram-control FILE] [--state FILE] "
                 "[--resume FILE]\n");
}

// sd-scripts' truncated logit-normal (flux_train_utils, sample_truncated_logit_normal_indices): S = sigmoid(-N),
// truncated at `cutoff`, stretched by 1 / cutoff, snapped to the T-point grid, redrawn uniform with prob `uniform`.
// `sigmax` (0 = off) redraws a shifted sigma past it (an earlier trainer's losses went non-finite past ~0.955;
// 2026-11-10's, on the RX 7900 XTX, stays finite to 1.0).
float sample_sigma(std::mt19937_64& rng, float logit_mean, float logit_std, float cutoff, float uniform,
                   float sigmax = 0.f) {
    std::normal_distribution<float> gauss(0.f, 1.f);
    std::uniform_real_distribution<float> uni(0.f, 1.f);
    float s;
    if (uni(rng) < uniform) {
        s = uni(rng);
    } else {
        const float u = gauss(rng);
        s = 1.f / (1.f + std::exp(logit_mean + logit_std * u));   // sigmoid(-(mean + std * u))
        if (s > cutoff) return sample_sigma(rng, logit_mean, logit_std, cutoff, 0.f, sigmax);   // redraw, no second uniform
        s /= cutoff;
    }
    s = std::ceil(s * 1000.f) / 1000.f;   // the 1000-point grid
    const float shift = 6.0f;             // Mage-Flow's static shift (MF_SHIFT)
    const float sig = shift * s / (1.f + (shift - 1.f) * s);
    if (sigmax > 0.f && sig > sigmax)
        return sample_sigma(rng, logit_mean, logit_std, cutoff, 0.f, sigmax);   // redraw past the broken tail
    return sig;
}

// Mage-Flow's reference trainer's default (ai-toolkit's flow-match "sigmoid" timesteps): sigma = sigmoid(-N(0, 1)),
// no shift, median 0.5.  The shifted draw above puts its median near 0.96, where a step sees almost nothing but noise
// and a character's details are not learned (2026-11-09: a base-model run left the images as they were)
float sample_sigma_logit_normal(std::mt19937_64& rng) {
    std::normal_distribution<float> gauss(0.f, 1.f);
    return 1.f / (1.f + std::exp(gauss(rng)));
}

}  // namespace

int main(int argc, char** argv) {
    strata::image::train::install_stop_signals();   // stop at a boundary and unwind the GPU owners
    if (argc < 2) {
        usage();
        return 2;
    }
    const std::string cmd = argv[1];
    std::string dit, data, stem, method = "lora", out, trigger, title, revision = "card-5";
    std::string vram_control, state_path, resume_path;   // the VRAM control (train::Control)
    long long vram_budget = 0;
    std::string train_blocks, only_if, ignore_if, kind;   // train::Targets' values, and the metadata's kind
    // ai-toolkit's Mage-Flow defaults (its mageflow model's job): linear timesteps, rank 32 with alpha = rank,
    // AdamW lr 1e-4 eps 1e-6 weight decay 1e-4 at a constant rate, batch 1, 3,000 steps, a save every 250, the
    // caption dropout from the manifest (the preprocessing default is the shared 0). "logit-normal" and "shifted"
    // are the earlier SPEC's draws.
    std::string timesteps = "linear";
    float sigma = 0.5f, lr = 1e-4f, alpha = 0.f, sigma_max = 0.f, adam_eps = 1e-6f, wd = 1e-4f, dropout = -1.f;
    int rank = 32, steps = 3000, accum = 1, checkpoint = 250, seed = 12345, warmup = 0;
    bool mod = true;
    std::string base = "int8";   // ai-toolkit quantizes Mage-Flow's transformer (qfloat8; XeStrata's QLoRA is int8)   // ai-toolkit's targets take the modulation linears (img_mod.1, txt_mod.1) as well
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "mf-train: %s needs a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--vram-budget")
            vram_budget = std::stoll(val("vram-budget"));
        else if (a == "--vram-control")
            vram_control = val("vram-control");
        else if (a == "--state")
            state_path = val("state");
        else if (a == "--resume")
            resume_path = val("resume");
        else if (a == "--dit")
            dit = val("dit");
        else if (a == "--data")
            data = val("data");
        else if (a == "--stem")
            stem = val("stem");
        else if (a == "--sigma")
            sigma = std::stof(val("sigma"));
        else if (a == "--seed")
            seed = std::stoi(val("seed"));
        else if (a == "--method")
            method = val("method");
        else if (a == "--rank")
            rank = std::stoi(val("rank"));
        else if (a == "--alpha")
            alpha = std::stof(val("alpha"));
        else if (a == "--steps")
            steps = std::stoi(val("steps"));
        else if (a == "--lr")
            lr = std::stof(val("lr"));
        else if (a == "--out")
            out = val("out");
        else if (a == "--accum")
            accum = std::stoi(val("accum"));
        else if (a == "--checkpoint")
            checkpoint = std::stoi(val("checkpoint"));
        else if (a == "--trigger")
            trigger = val("trigger");
        else if (a == "--sigma-max")
            sigma_max = std::stof(val("sigma-max"));
        else if (a == "--timesteps")
            timesteps = val("timesteps");
        else if (a == "--title")
            title = val("title");
        else if (a == "--revision")
            revision = val("revision");
        else if (a == "--caption-dropout")
            dropout = std::stof(val("caption-dropout"));
        else if (a == "--base")
            base = val("base");
        else if (a == "--mod")
            mod = val("mod") != "0";
        else if (a == "--warmup")
            warmup = std::stoi(val("warmup"));
        else if (a == "--eps")
            adam_eps = std::stof(val("eps"));
        else if (a == "--weight-decay")
            wd = std::stof(val("weight-decay"));
        else if (a == "--train-blocks")
            train_blocks = val("train-blocks");
        else if (a == "--only-if-contains")
            only_if = val("only-if-contains");
        else if (a == "--ignore-if-contains")
            ignore_if = val("ignore-if-contains");
        else if (a == "--kind")
            kind = val("kind");
        else {
            usage();
            return 2;
        }
    }
    if (dit.empty()) {
        usage();
        return 2;
    }
    void* s = strata::core::Runtime::get().create_stream();
    std::string err;
    const int64_t open_rank = (cmd == "train") ? rank : 0;
    if (base != "int8" && base != "bf16") {
        usage();
        return 2;
    }
    strata::image::train::Targets targets;   // read before the DiT is opened: a mistyped value is said at once
    if (!strata::image::train::parse_targets(train_blocks, only_if, ignore_if, targets, err) ||
        !strata::image::train::lora_kind_ok(kind)) {
        std::fprintf(stderr, "mf-train: %s\n", err.empty() ? "--kind is character, style or other" : err.c_str());
        return 2;
    }
    if (cmd == "train" && base == "int8" && method == "dora") {
        std::fprintf(stderr, "mf-train: DoRA reads the base's row norms: train it on --base bf16\n");
        return 2;
    }
    std::unique_ptr<strata::image::MfTrainModel, void (*)(strata::image::MfTrainModel*)> m(
        strata::image::mf_trainer_open(dit, open_rank, s, err, cmd == "train" && base == "int8"), strata::image::mf_trainer_close);
    if (m == nullptr) {
        std::fprintf(stderr, "mf-train: %s\n", err.c_str());
        return 1;
    }
    if (cmd == "info") {
        std::printf("dit: C %lld F %lld heads %lld head_dim %lld blocks %lld txt_dim %lld in_ch %lld\n",
                    (long long) m->C, (long long) m->F, (long long) m->heads, (long long) m->hd,
                    (long long) m->nblk, (long long) m->txt_dim, (long long) m->in_ch);
        return 0;
    }
    if (cmd == "train") {
        if (data.empty() || out.empty() || (method != "lora" && method != "lokr" && method != "dora") || rank <= 0) {
            usage();
            return 2;
        }
        try {
            // the manifest's stems and caption-dropout rates (skip the header's comment lines).
            // the rate column is the manifest's 10th tab field (SPEC v1 default 0.0; card-3 data carry 0).
            // a manifest line without the rate column is refused, never treated as 0.
            std::vector<std::string> stems;
            std::vector<float> drop_rates;
            {
                std::ifstream mf(data + "/manifest.tsv");
                std::string line;
                while (std::getline(mf, line)) {
                    if (line.empty() || line[0] == '#') continue;
                    std::vector<std::string> cols;
                    {
                        std::stringstream ls(line);
                        std::string tok;
                        while (std::getline(ls, tok, '\t')) cols.push_back(tok);
                    }
                    if (cols.size() < 10)
                        throw std::runtime_error("manifest line has no caption_dropout_rate column: " +
                                                 cols[0]);
                    float dr = 0;
                    try {
                        dr = std::stof(cols[9]);
                    } catch (...) {
                        throw std::runtime_error("manifest line has no caption_dropout_rate column: " +
                                                 cols[0]);
                    }
                    if (!(dr >= 0.f && dr <= 1.f))
                        throw std::runtime_error("caption_dropout_rate out of [0,1]: " + cols[0]);
                    stems.push_back(cols[0]);
                    drop_rates.push_back(dr);
                }
            }
            if (stems.empty()) throw std::runtime_error("no samples in " + data);
            if (timesteps != "linear" && timesteps != "logit-normal" && timesteps != "shifted")
                throw std::runtime_error("--timesteps is linear, logit-normal or shifted, not " + timesteps);
            if (alpha <= 0.f) alpha = (float) rank;   // ai-toolkit's alpha = rank (a scale of 1)
            if (mod && m->mrank > 0) {   // MageTrail: the engine adds no adapter to its low-rank modulation
                std::printf("train: MageTrail's low-rank modulation is not a target\n");
                mod = false;
            }
            if (dropout >= 0.f) {   // the flag decides for every sample over the manifest's column
                if (dropout > 1.f) throw std::runtime_error("--caption-dropout out of [0,1]");
                for (float& d : drop_rates) d = dropout;
            }
            // ai-toolkit's linear timesteps: torch.linspace(1000, 1, 1000)[randint(0, 999)], sigma = t / 1000
            const strata::image::train::Schedule lin = strata::image::train::schedule_linear();
            std::printf("train: %zu samples method %s rank %d alpha %.1f steps %d lr %.1e seed %d accum %d dropout %.3g "
                        "timesteps %s sigmamax %.3g base %s mod %d\n",
                        stems.size(), method.c_str(), rank, (double) alpha, steps, (double) lr, seed, accum,
                        (double) drop_rates[0], timesteps.c_str(), (double) sigma_max, base.c_str(), (int) mod);
            std::mt19937_64 rng(seed);
            std::normal_distribution<float> gauss(0.f, 1.f);
            std::uniform_real_distribution<float> drop_uni(0.f, 1.f);
            strata::image::mf_trainer_init_adapters(m.get(), method == "dora" ? 1 : method == "lokr" ? 2 : 0, alpha, seed,
                                                    mod, targets);
            strata::prefill::Gemm gemm;
            if (!gemm.init(s, 1 << 24, err)) throw std::runtime_error(err);
            // one sample's shapes set the device buffers; buckets differ, so re-read per step (cheap host work)
            // the export carries the empty prompt's rows beside the prompt rows; the manifest's rate decides.
            double acc_loss = 0;
            int acc_n = 0, nskip = 0, updates = 0;
            // fixed eval: the same stems and sigmas at start and end, so the two losses compare.
            // 8 stems max, sigma 0.5, eps from seed 9999 + index (deterministic, no training rng).
            const size_t eval_n = std::min<size_t>(8, stems.size());
            std::vector<std::string> eval_stems(stems.begin(), stems.begin() + (std::ptrdiff_t) eval_n);
            auto eval_avg = [&](const char* tag) -> float {
                double sum = 0;
                for (size_t ei = 0; ei < eval_n; ++ei) {
                    strata::image::MfTrainSample es =
                        strata::image::mf_trainer_sample(data, eval_stems[ei], m->txt_dim);
                    std::mt19937_64 erng((uint64_t) 9999 + ei);
                    std::normal_distribution<float> egauss(0.f, 1.f);
                    std::vector<float> eeps((size_t) es.tokens * 128);
                    for (float& v : eeps) v = egauss(erng);
                    strata::image::MfTrainCond ec =
                        strata::image::mf_trainer_cond(m.get(), 0.5f, es.h, es.w, gemm);
                    std::vector<float> vp((size_t) es.tokens * 128);
                    void* dv = nullptr;
                    if (!strata::gpu::alloc_device(&dv, (size_t) vp.size() * 4))
                        throw std::runtime_error("alloc");
                    const float eloss = strata::image::mf_trainer_forward(
                        m.get(), es, ec, eeps.data(), 0.5f, static_cast<float*>(dv), nullptr, gemm);
                    strata::gpu::free(dv);
                    strata::image::mf_trainer_free_cond(m.get(), ec);
                    if (!std::isfinite(eloss)) throw std::runtime_error("non-finite fixed eval loss");
                    sum += eloss;
                }
                const float avg = (float) (sum / (double) eval_n);
                std::printf("eval %s loss %.6f (fixed %zu stems sigma 0.5)\n", tag, (double) avg, eval_n);
                std::fflush(stdout);
                return avg;
            };
            std::vector<size_t> order(stems.size());
            for (size_t i = 0; i < order.size(); ++i) order[i] = i;
            // the VRAM control: the budget, pause/resume, the state file (the loop's micro-step, updates, rng, order)
            int micro = 0;
            auto save_loop = [&](std::ostream& os) {
                os << micro << '\n' << updates << '\n' << nskip << '\n' << acc_n << '\n' << acc_loss << '\n'
                   << rng << '\n' << gauss << '\n' << order.size() << '\n';
                for (size_t v : order) os << v << ' ';
                os << '\n';
            };
            auto load_loop = [&](std::istream& is) {
                size_t on = 0;
                is >> micro >> updates >> nskip >> acc_n >> acc_loss >> rng >> gauss >> on;
                order.resize(on);
                for (size_t& v : order) is >> v;
                if (!is) throw std::runtime_error("the state file's loop state did not read");
            };
            strata::image::train::Control::Opts co;
            co.path = vram_control;
            co.budget_mib = (uint64_t) vram_budget;
            co.state = state_path.empty() ? out + ".state.bin" : state_path;
            co.initial = strata::image::train::initial_vram(m->mrank > 0 ? "magetrail" : "mage-flow", base,
                                                                       strata::image::mf_train_max_tokens(data), rank);
            strata::image::train::Control ctl(strata::image::mf_trainer_base(m.get()), co);
            ctl.load(resume_path, load_loop);
            strata::image::train::log_training_plan(strata::image::mf_trainer_base(m.get()), lr, std::max(0, warmup));
            ctl.report();
            if (!ctl.boundary(true, save_loop)) return 0;   // initial eval also needs the largest cached shape's budget
            eval_avg("init");
            for (int step = micro; step < steps; ++step) {
                micro = step;
                if (!ctl.boundary(acc_n == 0, save_loop)) return 0;   // the state includes any accumulated gradients
                if ((size_t)(step % stems.size()) == 0)
                    std::shuffle(order.begin(), order.end(), rng);   // a shuffled epoch
                strata::image::MfTrainSample smp =
                    strata::image::mf_trainer_sample(data, stems[order[(size_t) step % order.size()]],
                                                     m->txt_dim);
                // caption dropout: the manifest's rate column decides per sample (SPEC v1 default 0.0).
                // the draw is always consumed so the rng stream stays identical at any rate; at 0 it never swaps.
                const float dr = drop_rates[order[(size_t) step % order.size()]];
                if (drop_uni(rng) < dr) {
                    smp.te = smp.uncond;
                    smp.rows = smp.uncond_rows;
                }
                // the step's latent drawn from the posterior (ai-toolkit samples latent_dist each step)
                if (smp.lat_sd.size() == smp.lat.size())
                    for (size_t i = 0; i < smp.lat.size(); ++i) smp.lat[i] += smp.lat_sd[i] * gauss(rng);
                const float sig =
                    timesteps == "linear"
                        ? lin.t[(size_t) strata::image::train::draw_index(rng, strata::image::train::IndexDraw::balanced)] /
                              1000.f
                    : timesteps == "shifted" ? sample_sigma(rng, -1.09f, 0.95f, 0.95f, 0.01f, sigma_max)
                                             : sample_sigma_logit_normal(rng);
                std::vector<float> eps((size_t) smp.tokens * 128);
                for (float& v : eps) v = gauss(rng);
                strata::image::MfTrainCond c = strata::image::mf_trainer_cond(m.get(), sig, smp.h, smp.w, gemm);
                strata::image::MfTrainState stt;
                const float loss = strata::image::mf_trainer_forward_keep(m.get(), smp, c, eps.data(), sig,
                                                                          stt, gemm);
                if (!std::isfinite(loss)) {
                    // a NaN loss would poison AdamW, so the micro-step is dropped and recorded instead
                    std::fprintf(stderr, "skip micro %d stem %s sigma %.4f loss %f (non-finite, no update)\n",
                                 step, stems[order[(size_t) step % order.size()]].c_str(), (double) sig,
                                 (double) loss);
                    strata::image::mf_trainer_free_state(m.get(), stt);
                    strata::image::mf_trainer_free_cond(m.get(), c);
                    ++nskip;
                    continue;
                }
                strata::image::mf_trainer_backward(m.get(), smp, c, stt, sig, gemm);
                strata::image::mf_trainer_free_state(m.get(), stt);
                strata::image::mf_trainer_free_cond(m.get(), c);
                ctl.report();
                acc_loss += loss;
                if (++acc_n < accum) continue;
                acc_n = 0;
                ++updates;
                const float lrw = warmup > 0 ? lr * std::min(1.f, (float) updates / (float) warmup) : lr;
                strata::image::mf_trainer_adamw(m.get(), lrw, 0.9f, 0.999f, adam_eps, wd, accum);
                const double update_loss = acc_loss / accum;
                if (updates % 10 == 0 || updates == 1) {
                    std::printf("step %5d loss %.6f lr %.1e\n", updates, acc_loss / accum, (double) lrw);
                    std::fflush(stdout);
                }
                acc_loss = 0;
                ctl.report();
                if (updates % checkpoint == 0)
                    strata::image::mf_trainer_save(m.get(), out + "." + std::to_string(updates) + ".safetensors",
                                                   title, trigger, seed, lr, updates, steps, kind);
                ctl.progress(updates, (steps + accum - 1) / accum, update_loss);
            }
            if (acc_n > 0) {   // the tail micro-steps' gradients, averaged over what ran
                strata::image::mf_trainer_adamw(m.get(), lr, 0.9f, 0.999f, adam_eps, wd, acc_n);
                ++updates;
                ctl.progress(updates, (steps + accum - 1) / accum, acc_loss / acc_n);
            }
            acc_n = 0;
            acc_loss = 0;
            micro = steps;
            if (!ctl.boundary(true, save_loop)) return 0;   // a save_state asked at the tail: written here, then the pause
            eval_avg("final");
            strata::image::mf_trainer_save(m.get(), out + ".final.safetensors", title, trigger, seed, lr, updates,
                                           steps, kind);
            ctl.save(save_loop);
            std::printf("done: saved %s (skipped %d non-finite micro-steps)\n",
                        (out + ".final.safetensors").c_str(), nskip);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "mf-train: %s\n", e.what());
            return 1;
        }
        return 0;
    }
    if (cmd != "loss" || data.empty() || stem.empty()) {
        usage();
        return 2;
    }
    try {
        strata::image::MfTrainSample smp = strata::image::mf_trainer_sample(data, stem, m->txt_dim);
        std::printf("sample: tokens %lld text rows %lld (%lld x %lld)\n", (long long) smp.tokens,
                    (long long) smp.rows, (long long) smp.w, (long long) smp.h);
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> gauss(0.f, 1.f);
        std::vector<float> eps((size_t) smp.tokens * 128);
        for (float& v : eps) v = gauss(rng);
        strata::prefill::Gemm gemm;
        if (!gemm.init(s, 1 << 24, err)) throw std::runtime_error(err);
        if (vram_budget > 0)
            strata::image::mf_trainer_base(m.get()).set_vram_budget((uint64_t) vram_budget << 20);
        strata::image::MfTrainCond c = strata::image::mf_trainer_cond(m.get(), sigma, smp.h, smp.w, gemm);
        std::vector<float> vp((size_t) smp.tokens * 128);
        void* dv = nullptr;
        if (!strata::gpu::alloc_device(&dv, (size_t) vp.size() * 4)) throw std::runtime_error("alloc");
        const float loss = strata::image::mf_trainer_forward(m.get(), smp, c, eps.data(), sigma,
                                                             static_cast<float*>(dv), nullptr, gemm);
        strata::gpu::free(dv);
        strata::image::mf_trainer_free_cond(m.get(), c);
        std::printf("loss: %.6f (sigma %.3f seed %llu)\n", (double) loss, (double) sigma,
                    (unsigned long long) seed);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mf-train: %s\n", e.what());
        return 1;
    }
    return 0;
}
