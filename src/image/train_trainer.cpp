// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/train_trainer.cpp - see include/strata/image/train_trainer.hpp.
#include "strata/image/train_trainer.hpp"

#include "strata/core/gpu.hpp"
#include "strata/platform/env.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace strata::image::train {

namespace {

constexpr uint64_t kTransients = uint64_t{3} << 29;   // 1.5 GiB beside a step's own buffers
constexpr uint64_t kFirstStep = uint64_t{3} << 30;    // 3 GiB more before the first step

}  // namespace

Trainer::Trainer(std::string what) : what_(std::move(what)) {}

Trainer::~Trainer() {
    if (s != nullptr) gpu::stream_sync(s);
    ws.drop();
    drop_tapes();
    for (void* p : owned_) gpu::free(p);
}

bool Trainer::init(void* stream, size_t ws_bytes, std::string& err) {
    s = stream;
    return gemm.init(stream, (int64_t) ws_bytes, err);
}

void* Trainer::take(size_t bytes) {
    void* p = nullptr;
    if (!gpu::alloc_device(&p, bytes))
        throw std::runtime_error("image engine (" + what_ + "): GPU memory (" + std::to_string(bytes >> 20) +
                                 " MiB): " + gpu::last_error());
    owned_.push_back(p);
    if (!gpu::memset_async(p, 0, bytes, s) || !gpu::stream_sync(s))
        throw std::runtime_error("image engine (" + what_ + "): zero");
    return p;
}

void Trainer::release(void* p) {
    if (p == nullptr) return;
    gpu::stream_sync(s);
    owned_.erase(std::remove(owned_.begin(), owned_.end(), p), owned_.end());
    gpu::free(p);
}

float* Trainer::dev_vec(const std::vector<float>& v) {
    float* d = get<float>((int64_t) v.size());
    if (!gpu::copy(d, v.data(), v.size() * 4)) throw std::runtime_error("image engine (" + what_ + "): copy");
    return d;
}

uint16_t* Trainer::dev16(const void* v, int64_t n) {
    uint16_t* d = get<uint16_t>(n);
    if (!gpu::copy(d, v, (size_t) n * 2)) throw std::runtime_error("image engine (" + what_ + "): copy");
    return d;
}

std::vector<float> Trainer::read(const float* p, int64_t n) const {
    std::vector<float> v((size_t) n);
    gpu::stream_sync(s);
    if (!gpu::copy(v.data(), p, (size_t) n * 4)) throw std::runtime_error("image engine (" + what_ + "): a read back");
    return v;
}

Lora& Trainer::add_lora(int64_t N, int64_t K, int64_t rank, std::mt19937_64& rng, bool views) {
    Lora& l = loras_.emplace_back();
    l.N = N;
    l.K = K;
    l.views = views;
    std::vector<float> A;
    lora_init(rng, rank, K, A);
    l.A = dev_vec(A);
    l.B = get<float>(N * rank);
    l.ad.rank = rank;
    l.ad.scale = 1.0f;   // alpha = rank
    l.ad.dA = get<float>(rank * K);
    l.ad.dB = get<float>(N * rank);
    add_param(l.A, l.ad.dA, rank * K);
    add_param(l.B, l.ad.dB, N * rank);
    if (views) {
        l.ad.a = get<uint16_t>(rank * K);
        l.ad.b = get<uint16_t>(N * rank);
        if (!gemm.f16_batch_ex_ok()) {   // the transposed views, where the products cannot read the rows
            l.ad.at = get<uint16_t>(rank * K);
            l.ad.bt = get<uint16_t>(N * rank);
        }
        sync_views(l.ad, l.A, l.B, N, K);
    }
    return l;
}

void Trainer::add_param(float* w, float* g, int64_t n) {
    params_.push_back({w, g, get<float>(n), get<float>(n), n});
}

void Trainer::sync_views(const Adapter& ad, const float* A, const float* B, int64_t N, int64_t K) {
    const int64_t r = ad.rank;
    f16_copy(A, r * K, ad.a, s);
    if (ad.at != nullptr) transpose_f16(ad.a, r, K, ad.at, s);
    f16_copy(B, N * r, ad.b, s);
    if (ad.bt != nullptr) transpose_f16(ad.b, N, r, ad.bt, s);
}

double Trainer::update(const AdamW& o, float gscale) {
    std::vector<std::pair<const float*, int64_t>> parts;
    parts.reserve(params_.size());
    for (const Param& p : params_) parts.emplace_back(p.g, p.n);
    const double norm = std::sqrt(sq_sum(parts, s)) * (double) gscale;
    const float clip = (float) std::min(1.0, 1.0 / (norm + 1e-6));   // torch's clip_grad_norm_ at 1
    ++steps;
    for (const Param& p : params_) adamw(p.w, p.g, p.m, p.v, p.n, o, steps, gscale * clip, s);
    for (const Lora& l : loras_)
        if (l.views) sync_views(l.ad, l.A, l.B, l.N, l.K);
    gpu::stream_sync(s);
    return norm;
}

void Trainer::dgrad_scratch(const Product* p, int n, bool int8) {
    const bool rows = gemm.f16_batch_ex_ok();
    dgrad_tmp_.assign((size_t) n, nullptr);
    int64_t most = 0;
    for (int i = 0; i < n; ++i) {
        if (!rows || int8) dgrad_tmp_[(size_t) i] = get<uint16_t>(p[i].N * p[i].K);
        most = std::max(most, p[i].N * p[i].K);
    }
    if (!rows && int8) w8tmp_ = get<uint16_t>(most);
}

void Trainer::dgrad_weights(Product* p, int n) {
    const bool rows = gemm.f16_batch_ex_ok();
    for (int i = 0; i < n; ++i) {
        Product& pr = p[i];
        uint16_t* tmp = dgrad_tmp_.at((size_t) i);
        if (rows) {
            if (pr.w8 != nullptr) w8_dequant_rows(pr.w8, pr.sw, pr.N, pr.K, tmp, s);
            pr.wr = pr.w8 != nullptr ? tmp : pr.w;
            pr.wt = nullptr;
            continue;
        }
        if (pr.w8 != nullptr) w8_dequant_t(pr.w8, pr.sw, pr.N, pr.K, w8tmp_, tmp, s);
        else transpose_f16(pr.w, pr.N, pr.K, tmp, s);
        pr.wt = tmp;
        pr.wr = nullptr;
    }
}

void Trainer::set_budget(int64_t rows, uint64_t keep_row, uint64_t guess_row) {
    if (rows <= budget_rows_ && budget_stepped_ == stepped && ws.vram() > 0) return;
    budget_rows_ = std::max(budget_rows_, rows);
    budget_stepped_ = stepped;
    const int64_t R = budget_rows_;
    const uint64_t all = ws.total();
    uint64_t budget = all;
    size_t free_b = 0, total = 0;
    gpu::stream_sync(s);
    const bool known = gpu::mem_info(&free_b, &total);
    if (const char* e = strata::env_get("TRAIN_WEIGHTS_MIB")) {
        budget = (uint64_t) std::strtoll(e, nullptr, 10) << 20;
    } else if (known) {
        const uint64_t have = (uint64_t) free_b + ws.vram() + tape_cap_;
        const uint64_t reserve = keep_row * (uint64_t) R +
                                 scratch_growth(arena.held(), R, scratch_rows, guess_row * (uint64_t) R) +
                                 kTransients + (stepped ? 0 : kFirstStep);
        budget = have > reserve ? have - reserve : 0;
    }
    budget = std::min(all, std::max(budget, ws.min_budget()));
    if (budget < all)
        std::fprintf(stderr, "image engine (%s): the blocks' weights %llu of %llu MiB in VRAM, the rest streamed from "
                             "RAM\n",
                     what_.c_str(), (unsigned long long) (budget >> 20), (unsigned long long) (all >> 20));
    if (tape_cap_ > 0 && known && budget > (uint64_t) free_b + ws.vram()) drop_tapes();   // the weights take theirs
    ws.configure(budget, s);
}

bool Trainer::inputs_half(int64_t rows, uint64_t f32_bytes, uint64_t guess_row) {
    if (const char* e = strata::env_get("TRAIN_INPUTS16")) return std::strtol(e, nullptr, 10) != 0;
    size_t free_b = 0, total = 0;
    gpu::stream_sync(s);
    if (!gpu::mem_info(&free_b, &total)) return false;
    const uint64_t have = (uint64_t) free_b + tape_cap_;
    const uint64_t need = f32_bytes + scratch_growth(arena.held(), rows, scratch_rows, guess_row * (uint64_t) rows) +
                          kTransients + (stepped ? 0 : kFirstStep);
    return have < need;
}

int64_t Trainer::keep_tapes(int64_t nb, int64_t rows, size_t per, size_t step_bytes) {
    if (tapes_.size() != (size_t) nb) tapes_.resize((size_t) nb);
    per = (per + 255) / 256 * 256;
    gpu::stream_sync(s);
    size_t avail = 0;   // what the tapes may take this step
    if (const char* e = strata::env_get("TRAIN_TAPES")) {
        avail = (size_t) std::clamp<int64_t>(std::strtoll(e, nullptr, 10), 0, nb) * per;
    } else if (scratch_rows > 0 && rows <= scratch_rows) {
        size_t free_b = 0, total = 0;
        if (gpu::mem_info(&free_b, &total)) {
            free_b += tape_cap_;
            const size_t reserve = (size_t) scratch_growth(arena.held(), rows, scratch_rows, 0) + step_bytes +
                                   (size_t) kTransients;
            avail = free_b > reserve ? free_b - reserve : 0;
        }
    }
    const size_t want = std::min(avail, (size_t) nb * per);
    if (want / per > tape_cap_ / per || avail < tape_cap_) {
        drop_tapes();
        tapes_.resize((size_t) nb);
        if (want >= per && gpu::alloc_device(&tape_buf_, want)) tape_cap_ = want;
    }
    const int64_t k = std::min<int64_t>(nb, (int64_t) (tape_cap_ / per));
    if (strata::env_get("TRAIN_PROFILE") != nullptr) {
        size_t fb = 0, tb = 0;
        gpu::mem_info(&fb, &tb);
        std::fprintf(stderr, "kept tapes %lld of %lld (%zu MiB each, %zu MiB free of %zu)\n", (long long) k,
                     (long long) nb, per >> 20, fb >> 20, tb >> 20);
    }
    for (int64_t b = 0; b < nb; ++b) {
        auto& t = tapes_[(size_t) b];
        if (b >= nb - k) {
            if (!t) t = std::make_unique<Arena>();
            t->attach(static_cast<uint8_t*>(tape_buf_) + (size_t) (b - (nb - k)) * per, per);
        } else {
            t.reset();
        }
    }
    return k;
}

Arena* Trainer::tape(int64_t b) const {
    return b >= 0 && (size_t) b < tapes_.size() ? tapes_[(size_t) b].get() : nullptr;
}

void Trainer::stepped_rows(int64_t rows) {
    scratch_rows = std::max(scratch_rows, rows);
    stepped = true;
}

void Trainer::drop_tapes() {
    for (auto& t : tapes_) t.reset();
    if (tape_buf_ != nullptr) {
        gpu::stream_sync(s);
        gpu::free(tape_buf_);
    }
    tape_buf_ = nullptr;
    tape_cap_ = 0;
}

}  // namespace strata::image::train
