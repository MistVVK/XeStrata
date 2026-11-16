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
#include <istream>
#include <ostream>
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
    for (auto& [p, n] : owned_) gpu::free(p);
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
    owned_.emplace_back(p, bytes);
    owned_bytes_ += bytes;
    if (!gpu::memset_async(p, 0, bytes, s) || !gpu::stream_sync(s))
        throw std::runtime_error("image engine (" + what_ + "): zero");
    return p;
}

void Trainer::release(void* p) {
    if (p == nullptr) return;
    gpu::stream_sync(s);
    for (auto it = owned_.begin(); it != owned_.end(); ++it)
        if (it->first == p) {
            owned_bytes_ -= it->second;
            owned_.erase(it);
            break;
        }
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
    if (rows <= budget_rows_ && budget_stepped_ == stepped && ws.vram() > 0 && !replan_) return;
    budget_rows_ = std::max(budget_rows_, rows);
    budget_stepped_ = stepped;
    replan_ = false;
    if (guess_row > 0) guess_row_ = guess_row;
    const int64_t R = budget_rows_;
    const uint64_t all = ws.total();
    uint64_t budget = all;
    size_t free_b = 0, total = 0;
    gpu::stream_sync(s);
    const bool known = gpu::mem_info(&free_b, &total);
    if (const char* e = strata::env_get("TRAIN_WEIGHTS_MIB")) {
        budget = (uint64_t) std::strtoll(e, nullptr, 10) << 20;
    } else if (budget_ > 0) {
        // the explicit cap: what it leaves beside the trainer's own allocations and the arena, as the free
        // memory does in the automatic form (the step's buffers already taken are in owned_bytes_; the ones
        // this step will take are keep_row and the arena's growth)
        const uint64_t fixed = owned_bytes_ + arena.held();
        const uint64_t room = budget_ > fixed ? budget_ - fixed : 0;
        const uint64_t have = room + ws.vram() + tape_cap_;
        const uint64_t reserve = keep_row * (uint64_t) R +
                                 scratch_growth(arena.held(), R, scratch_rows, guess_row * (uint64_t) R) +
                                 kTransients + (stepped ? 0 : kFirstStep);
        budget = have > reserve ? have - reserve : 0;
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
    uint64_t have = 0;
    if (budget_ > 0) {
        const uint64_t fixed = owned_bytes_ + arena.held();
        have = (budget_ > fixed ? budget_ - fixed : 0) + tape_cap_;
    } else if (gpu::mem_info(&free_b, &total)) {
        have = (uint64_t) free_b + tape_cap_;
    } else {
        return false;
    }
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
        uint64_t room = 0;
        if (budget_ > 0) {
            const uint64_t fixed = owned_bytes_ + arena.held();
            room = budget_ > fixed ? budget_ - fixed : 0;
        } else if (gpu::mem_info(&free_b, &total)) {
            room = (uint64_t) free_b;
        }
        if (room > 0 || budget_ > 0) {
            room += tape_cap_;
            const size_t reserve = (size_t) scratch_growth(arena.held(), rows, scratch_rows, 0) + step_bytes +
                                   (size_t) kTransients;
            avail = room > reserve ? (size_t) (room - reserve) : 0;
        }
    }
    const size_t want = std::min(avail, (size_t) nb * per);
    if (want / per > tape_cap_ / per || avail < tape_cap_) {
        drop_tapes();
        tapes_.resize((size_t) nb);
        if (want >= per && gpu::alloc_device(&tape_buf_, want)) tape_cap_ = want;
    }
    const int64_t k = std::min<int64_t>(nb, (int64_t) (tape_cap_ / per));
    tapes_kept_ = k;
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

void Trainer::stepped_rows(int64_t rows, uint64_t row_bytes) {
    scratch_rows = std::max(scratch_rows, rows);
    last_rows_ = rows;
    if (row_bytes > 0) last_row_bytes_ = row_bytes;
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
    tapes_kept_ = 0;
}

// ---- the explicit budget, pause and resume

void Trainer::set_vram_budget(uint64_t bytes) {
    if (bytes == budget_) return;
    budget_ = bytes;
    replan_ = true;
}

uint64_t Trainer::held() const {
    return owned_bytes_ + arena.held() + tape_cap_ + ws.vram();
}

uint64_t Trainer::min_bytes(int64_t rows, uint64_t row_bytes, uint64_t guess_row) const {
    return owned_bytes_ + ws.min_budget() + 2 * row_bytes +
           scratch_growth(arena.held(), rows, scratch_rows, guess_row * (uint64_t) rows) + kTransients +
           (stepped ? 0 : kFirstStep);
}

uint64_t Trainer::est_bytes(int64_t rows, uint64_t guess_row) const {
    return held() + scratch_growth(arena.held(), rows, scratch_rows, guess_row * (uint64_t) rows) + kTransients +
           (stepped ? 0 : kFirstStep);
}

void Trainer::pause() {
    if (paused_) return;
    gpu::stream_sync(s);
    host.close();
    host_.clear();
    old_owned_.clear();
    moved_.clear();
    for (auto& [p, bytes] : owned_) {
        Hosted e{p, bytes, std::vector<uint8_t>(bytes)};
        if (!gpu::copy(e.data.data(), p, bytes))
            throw std::runtime_error("image engine (" + what_ + "): pause read back: " + gpu::last_error());
        old_owned_.push_back(p);
        host_.push_back(std::move(e));
        gpu::free(p);
    }
    owned_.clear();
    owned_bytes_ = 0;
    ws.drop();          // the regions go; the pinned homes stay (the resume copies from them)
    drop_tapes();
    arena.release(s);
    paused_ = true;
    replan_ = true;
}

void Trainer::resume() {
    if (!paused_) return;
    for (Hosted& e : host_) {
        void* p = gpu::alloc_device(e.bytes);
        if (p == nullptr)
            throw std::runtime_error("image engine (" + what_ + "): resume GPU memory (" +
                                     std::to_string(e.bytes >> 20) + " MiB): " + gpu::last_error());
        if (!gpu::copy(p, e.data.data(), e.bytes))
            throw std::runtime_error("image engine (" + what_ + "): resume copy: " + gpu::last_error());
        moved_[e.old] = p;
        owned_.emplace_back(p, e.bytes);
        owned_bytes_ += e.bytes;
        e.data.clear();
        e.data.shrink_to_fit();
    }
    auto mv = [&](auto*& p) {
        using T = std::remove_reference_t<decltype(p)>;
        p = static_cast<T>(moved(p));
    };
    for (Param& p : params_) {
        mv(p.w);
        mv(p.g);
        mv(p.m);
        mv(p.v);
    }
    for (Lora& l : loras_) {
        mv(l.A);
        mv(l.B);
        Adapter& a = l.ad;
        mv(a.a);
        mv(a.at);
        mv(a.b);
        mv(a.bt);
        mv(a.dA);
        mv(a.dB);
        mv(a.m);
        mv(a.norms);
        mv(a.dm);
        mv(a.w1);
        mv(a.w1t);
        mv(a.w2a);
        mv(a.w2b);
        mv(a.dW1);
        mv(a.dW2a);
        mv(a.dW2b);
    }
    for (uint16_t*& p : dgrad_tmp_) mv(p);
    mv(w8tmp_);
    remap();   // the model's own persistent pointers
    paused_ = false;
    replan_ = true;
}

void* Trainer::moved(void* old) const {
    if (old == nullptr) return nullptr;
    auto it = moved_.find(old);
    if (it != moved_.end()) return it->second;
    for (void* p : old_owned_)
        if (p == old)
            throw std::logic_error("image engine (" + what_ + "): a pointer not remapped after resume");
    return old;
}

void Trainer::save_state(std::ostream& os) const {
    gpu::stream_sync(s);
    const int64_t n = (int64_t) params_.size();
    os.write(reinterpret_cast<const char*>(&n), sizeof n);
    std::vector<float> h;
    for (const Param& p : params_) {
        os.write(reinterpret_cast<const char*>(&p.n), sizeof p.n);
        for (const float* d : {p.w, p.m, p.v}) {
            h.resize((size_t) p.n);
            if (!gpu::copy(h.data(), d, (size_t) p.n * 4))
                throw std::runtime_error("image engine (" + what_ + "): state read back");
            os.write(reinterpret_cast<const char*>(h.data()), (std::streamsize) (p.n * 4));
        }
    }
}

void Trainer::load_state(std::istream& is) {
    int64_t n = 0;
    is.read(reinterpret_cast<char*>(&n), sizeof n);
    if (n != (int64_t) params_.size()) throw std::runtime_error("image engine (" + what_ + "): state: parameters");
    std::vector<float> h;
    for (Param& p : params_) {
        int64_t pn = 0;
        is.read(reinterpret_cast<char*>(&pn), sizeof pn);
        if (pn != p.n) throw std::runtime_error("image engine (" + what_ + "): state: a parameter's size");
        h.resize((size_t) p.n);
        for (float* d : {p.w, p.m, p.v}) {
            is.read(reinterpret_cast<char*>(h.data()), (std::streamsize) (p.n * 4));
            if (!is) throw std::runtime_error("image engine (" + what_ + "): state: short");
            if (!gpu::copy(d, h.data(), (size_t) p.n * 4))
                throw std::runtime_error("image engine (" + what_ + "): state copy");
        }
        if (!gpu::memset_async(p.g, 0, (size_t) p.n * 4, s))
            throw std::runtime_error("image engine (" + what_ + "): state zero");
    }
    gpu::stream_sync(s);
    for (const Lora& l : loras_)
        if (l.views) sync_views(l.ad, l.A, l.B, l.N, l.K);
    gpu::stream_sync(s);
    replan_ = true;
}

// ---- HostInputs: the blocks' inputs on the host as FP32 (the values kept byte for byte)

bool Trainer::host_inputs(int64_t slots, int64_t row_bytes) {
    if (const char* e = strata::env_get("TRAIN_INPUTS16"))
        if (std::strtol(e, nullptr, 10) != 0) return false;
    if (const char* e = strata::env_get("TRAIN_INPUTS_HOST"))
        if (std::strtol(e, nullptr, 10) == 0) return false;
    return host.open(slots, row_bytes, s);
}

Trainer::HostInputs::~HostInputs() {
    close();
}

bool Trainer::HostInputs::open(int64_t slots, int64_t bytes, void* compute) {
    // the same form again (the next step): the pinned memory stays (pinning it every step costs more than a copy)
    if (h_ != nullptr && slot_bytes_ == bytes && (int64_t) done_.size() == slots && s_ == compute) return true;
    close();
    if (slots <= 0 || bytes <= 0) return false;
    // the pinned budget's headroom as the streamed layers' homes take it (qi_stream.cpp take_home): the RAM
    // left to the system and the other engines, not a fixed share of one machine's RAM
    uint8_t* h = static_cast<uint8_t*>(gpu::alloc_host((size_t) (slots * bytes)));
    if (h == nullptr) return false;
    gpu::Stream cs = gpu::stream_create_copy();
    if (cs == nullptr) {
        gpu::free(h);
        return false;
    }
    h_ = h;
    slot_bytes_ = bytes;
    s_ = compute;
    cs_ = cs;
    if (!gpu::event_create(&wr_) || !gpu::event_create(&used_[0]) || !gpu::event_create(&used_[1])) {
        close();
        return false;
    }
    done_.resize((size_t) slots, nullptr);
    for (auto& e : done_)
        if (!gpu::event_create(&e)) {
            close();
            return false;
        }
    kept_.resize((size_t) slots, nullptr);
    for (auto& e : kept_)
        if (!gpu::event_create(&e)) {
            close();
            return false;
        }
    return true;
}

void Trainer::HostInputs::close() {
    if (h_ == nullptr) return;
    gpu::stream_sync(s_);
    gpu::stream_sync(cs_);
    for (auto& e : done_)
        if (e != nullptr) gpu::event_destroy(e);
    done_.clear();
    for (auto& e : kept_)
        if (e != nullptr) gpu::event_destroy(e);
    kept_.clear();
    if (wr_ != nullptr) gpu::event_destroy(wr_);
    for (auto& e : used_)
        if (e != nullptr) gpu::event_destroy(e);
    wr_ = nullptr;
    used_[0] = used_[1] = nullptr;
    gpu::stream_destroy(cs_);
    gpu::free(h_);
    h_ = nullptr;
    cs_ = nullptr;
}

void Trainer::HostInputs::keep(int64_t slot, const float* src, int64_t bytes) {
    if (!gpu::event_record(wr_, s_) || !gpu::stream_wait_event(cs_, wr_) ||
        !gpu::copy_async(h_ + slot * slot_bytes_, src, (size_t) bytes, cs_, "the blocks' inputs to the host") ||
        !gpu::event_record(kept_[(size_t) slot], cs_))
        throw std::runtime_error("image engine: the inputs' keep: " + std::string(gpu::last_error()));
}

void Trainer::HostInputs::kept_wait(int64_t slot) {
    if (!gpu::stream_wait_event(s_, kept_[(size_t) slot]))
        throw std::runtime_error("image engine: the inputs' kept wait: " + std::string(gpu::last_error()));
}

void Trainer::HostInputs::fetch(int64_t slot, float* dst, int64_t bytes, int used_parity) {
    if (used_parity >= 0 && !gpu::stream_wait_event(cs_, used_[used_parity]))
        throw std::runtime_error("image engine: the inputs' fetch wait: " + std::string(gpu::last_error()));
    if (!gpu::copy_async(dst, h_ + slot * slot_bytes_, (size_t) bytes, cs_, "the blocks' inputs from the host") ||
        !gpu::event_record(done_[(size_t) slot], cs_))
        throw std::runtime_error("image engine: the inputs' fetch: " + std::string(gpu::last_error()));
}

void Trainer::HostInputs::wait(int64_t slot) {
    if (!gpu::stream_wait_event(s_, done_[(size_t) slot]))
        throw std::runtime_error("image engine: the inputs' wait: " + std::string(gpu::last_error()));
}

void Trainer::HostInputs::used(int p) {
    if (!gpu::event_record(used_[p], s_))
        throw std::runtime_error("image engine: the inputs' used: " + std::string(gpu::last_error()));
}

}  // namespace strata::image::train
