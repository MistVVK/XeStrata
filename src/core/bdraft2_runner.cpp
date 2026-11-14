// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// The buffers and stream of the YASAKANI block drafter.
#include "strata/core/bdraft2_runner.hpp"

#include "strata/core/on_device.hpp"
#include "strata/core/plain_model.hpp"
#include "strata/kernels/dflash2.hpp"
#include "strata/platform/env.hpp"

#include <algorithm>
#include <exception>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>

namespace strata::core {
namespace {
constexpr int B = 8, H = 5120, FF = 17408, Q = 4096, KV = 1024, SLOTS = 2064, RANK = 256;
}

Bdraft2Runner::~Bdraft2Runner() { release(); }

bool Bdraft2Runner::taps(const std::string& path, std::vector<int>& result, std::string& err) {
    try {
        const strata::yskn::Container c(path);
        if (c.model().arch != "bdraft2" || c.model().target_layers.size() != 5)
            throw std::runtime_error("bdraft2: invalid tap list");
        result = c.model().target_layers;
        return true;
    } catch (const std::exception& e) { err = e.what(); return false; }
}

void Bdraft2Runner::release() {
    if (device_ >= 0) {
        const OnDevice on(device_);
        if (stream_) gpu::stream_sync(stream_);
        weights_.release();
        for (void* p : allocations_) gpu::free(p);
    }
    allocations_.clear(); history_.reset(); bytes_ = 0; device_ = -1; model_ = nullptr; stream_ = nullptr;
    context_ = {}; block_ = {}; full_ = {}; full_bytes_ = 0; mode_ = Bdraft2Weights::HeadMode::Auto;
}

bool Bdraft2Runner::load(const std::string& path, const std::string& head, PlainModel& model, int device,
                         std::string& err) {
    release();
    device_ = device; model_ = &model; stream_ = model.stream();
    const OnDevice on(device_);
    try {
        const char* choice = strata::env_get("BDRAFT_SHEAD");
        const std::string mode = choice ? choice : "auto";
        if (mode != "auto" && mode != "on" && mode != "1" && mode != "off" && mode != "0")
            throw std::runtime_error("bdraft2: XESTRATA_BDRAFT_SHEAD must be auto, off or on");
        if (!weights_.load(path, mode == "off" || mode == "0" ? "" : head, device, (int) model.spec().n_embd,
                           (int) model.spec().n_layer, (int) model.vocab(), err)) throw std::runtime_error(err);
        if (mode == "on" || mode == "1") {
            if (!weights_.has_shead()) throw std::runtime_error("bdraft2: XESTRATA_BDRAFT_SHEAD=on needs a SHEAD file");
            set_head_mode(Bdraft2Weights::HeadMode::On);
        } else if (mode == "off" || mode == "0") {
            set_head_mode(Bdraft2Weights::HeadMode::Off);
        }
        auto alloc = [&](auto*& p, size_t n) {
            using T = std::remove_pointer_t<std::decay_t<decltype(p)>>;
            p = static_cast<T*>(gpu::alloc_device(n * sizeof(T)));
            if (!p) throw std::runtime_error("bdraft2: work buffers do not fit");
            allocations_.push_back(p); bytes_ += n * sizeof(T);
        };
        for (int l = 0; l < 5; ++l) { alloc(keys_[l], SLOTS * KV); alloc(values_[l], SLOTS * KV); }
        alloc(tags_, SLOTS); alloc(positions_, B); alloc(anchor_, 1); alloc(epoch_, 1); alloc(out_, B - 1);
        alloc(rope_, B * 128); alloc(anchor_embedding_, H); alloc(mask_embedding_, H); alloc(probabilities_, B - 1);
        alloc(context_.fc, B * H); alloc(context_.normalized, B * H); alloc(context_.key, B * KV);
        alloc(context_.value, B * KV); alloc(context_.quantized, B * H); alloc(context_.scales, B * H / 512);
        context_.projection = gpu::alloc_device(weights_.scratch_bytes());
        if (!context_.projection) throw std::runtime_error("bdraft2: projection scratch does not fit");
        allocations_.push_back(context_.projection); bytes_ += weights_.scratch_bytes();
        block_.projection = context_.projection;
        if (mode_ != Bdraft2Weights::HeadMode::On) {
            const auto* target_head = model.yskn_draft_head();
            if (!target_head) throw std::runtime_error("bdraft2: the target has no yskn output head");
            Bdraft2Weights::FullHead full;
            const uint64_t before = bytes_;
            full.weights = target_head;
            const size_t full_scratch = strata::kernels::yskn_scratch_bytes(*target_head, B - 1);
            full.projection = gpu::alloc_device(full_scratch);
            if (!full.projection) throw std::runtime_error("bdraft2: full head scratch does not fit");
            allocations_.push_back(full.projection); bytes_ += full_scratch;
            alloc(full.logits, (size_t) (B - 1) * model.vocab());
            alloc(full.top_scratch, (size_t) (B - 1) * kernels::kDf2TopBlocks * kernels::kDf2TopK * 2);
            weights_.set_full_head(full);
            full_ = full;
            full_bytes_ = bytes_ - before;
        }
        alloc(block_.h, B * H); alloc(block_.normalized, B * H); alloc(block_.conv, B * H);
        alloc(block_.dynamic, B * H / 4); alloc(block_.output, B * H);
        alloc(block_.q, B * Q); alloc(block_.qnorm, B * Q); alloc(block_.k, B * KV); alloc(block_.knorm, B * KV);
        alloc(block_.v, B * KV); alloc(block_.attention, B * Q);
        alloc(block_.gate, B * FF); alloc(block_.up, B * FF); alloc(block_.glu, B * FF); alloc(block_.final, (B - 1) * H);
        alloc(block_.hp, (B - 1) * RANK); alloc(block_.part, (size_t) kernels::kDf2PartChunks * 8 * 32 * 130);
        alloc(block_.scores, (B - 1) * 16); alloc(block_.ids, (B - 1) * 16);
        history_.init(H, 512, SLOTS, weights_.model().window);
        bool log_on = true;
        if (const char* e = strata::env_get("DF2_RING_LOG"); e && e[0] == '0') log_on = false;
        uint64_t budget = 256ull << 20;
        if (const char* e = strata::env_get("DF2_LOG_MIB"); e && *e)
            budget = (uint64_t) std::max(0, std::atoi(e)) << 20;
        history_.configure(log_on, budget);
        std::vector<float> mask(H);
        if (!model.embed_token(weights_.model().mask_token, mask.data()) ||
            !gpu::copy_async(mask_embedding_, mask.data(), H * sizeof(float), stream_, "bdraft2 mask") ||
            !gpu::memset_async(tags_, 0xff, SLOTS * sizeof(int64_t), stream_) || !epoch(history_.epoch(), err) ||
            !gpu::stream_sync(stream_)) throw std::runtime_error("bdraft2: initial buffers");
        return true;
    } catch (const std::exception& e) { err = e.what(); release(); return false; }
}

bool Bdraft2Runner::release_full_head(std::string& err) {
    if (!loaded() || !weights_.uses_shead()) { err = "bdraft2: the full head is in use"; return false; }
    const OnDevice on(device_);
    if (!gpu::stream_sync(stream_)) { err = "bdraft2: full head is still in use"; return false; }
    const std::initializer_list<void*> buffers = {full_.projection, full_.logits, full_.top_scratch};
    for (void* p : buffers)
        if (p && std::find(allocations_.begin(), allocations_.end(), p) == allocations_.end()) {
            err = "bdraft2: unknown full head allocation"; return false;
        }
    for (void* p : buffers) {
        if (!p) continue;
        gpu::free(p);
        allocations_.erase(std::find(allocations_.begin(), allocations_.end(), p));
    }
    bytes_ -= full_bytes_;
    full_ = {}; full_bytes_ = 0;
    weights_.set_full_head(full_);
    return true;
}

bool Bdraft2Runner::finish_head_choice(bool use_shead, std::string& err) {
    if (!loaded() || !head_auto()) { err = "bdraft2: the head choice is not automatic"; return false; }
    if (!gpu::stream_sync(stream_)) { err = "bdraft2: head choice synchronization"; return false; }
    weights_.set_probe_shead(use_shead);
    return use_shead ? release_full_head(err) : weights_.release_shead(err);
}

bool Bdraft2Runner::begin_request(const int32_t* ids, int64_t n, int64_t resume, bool* cold, std::string& err) {
    std::string replay_error;
    if (!begin_request(history_, ids, n, resume, cold, replay_error, err)) return false;
    if (!replay_error.empty())
        std::fprintf(stderr, "xestrata bdraft2: %s: the ring starts over for this request\n", replay_error.c_str());
    return true;
}

bool Bdraft2Runner::context(const float* taps, int64_t rows, int64_t pos0, std::string& err) {
    if (!loaded() || !taps || rows < 0 || pos0 < 0) { err = "bdraft2: invalid context"; return false; }
    try {
        for (int64_t r = 0; r < rows; r += B) {
            const int count = (int) std::min<int64_t>(B, rows - r);
            if (!commit_context(history_, taps + (size_t) r * 5 * H, count, pos0 + r, err)) return false;
        }
        return true;
    } catch (const std::exception& e) { err = e.what(); return false; }
}

bool Bdraft2Runner::draft(const float* taps, int rows, int64_t pos0, int32_t anchor, int32_t* drafts,
                          float* probabilities, std::string& err) {
    return context(taps, rows, pos0, err) && draft(anchor, pos0 + rows, drafts, probabilities, err);
}

bool Bdraft2Runner::epoch(int32_t value, std::string& err) {
    const OnDevice on(device_);
    if (!epoch_ || value < 0 || !gpu::copy_async(epoch_, &value, sizeof value, stream_, "bdraft2 epoch") ||
        !gpu::stream_sync(stream_)) { err = "bdraft2: epoch update"; return false; }
    return true;
}

bool Bdraft2Runner::stage_positions(const int32_t* positions, int rows, std::string& err) {
    std::vector<float> rope((size_t) rows * 128);
    kernels::df2_rope_table_cpu(positions, rows, weights_.model().rope_theta, rope.data());
    if (!gpu::copy_async(positions_, positions, (size_t) rows * sizeof(int32_t), stream_, "bdraft2 positions") ||
        !gpu::copy_async(rope_, rope.data(), rope.size() * sizeof(float), stream_, "bdraft2 rope") ||
        !gpu::stream_sync(stream_)) { gpu::stream_sync(stream_); err = "bdraft2: positions"; return false; }
    return true;
}

bool Bdraft2Runner::context(const float* taps, int rows, int64_t pos0, std::vector<int8_t>& q,
                            std::vector<float>& scales, std::string& err) {
    if (!loaded() || !taps || rows < 1 || rows > B || pos0 < 0 || pos0 > INT32_MAX - rows) {
        err = "bdraft2: invalid context"; return false;
    }
    try {
        const OnDevice on(device_);
        int32_t positions[B];
        for (int i = 0; i < rows; ++i) positions[i] = (int32_t) pos0 + i;
        if (!stage_positions(positions, rows, err)) return false;
        weights_.context(taps, rows, positions_, rope_, keys_, values_, SLOTS, tags_, epoch_, context_, stream_);
        q.resize((size_t) rows * H); scales.resize((size_t) rows * H / 512);
        if (!gpu::copy_async(q.data(), context_.quantized, q.size(), stream_, "bdraft2 log") ||
            !gpu::copy_async(scales.data(), context_.scales, scales.size() * sizeof(float), stream_, "bdraft2 log") ||
            !gpu::stream_sync(stream_)) throw std::runtime_error("bdraft2: context or log copy");
        return true;
    } catch (const std::exception& e) { gpu::stream_sync(stream_); err = e.what(); return false; }
}

bool Bdraft2Runner::replay(const int8_t* q, const float* scales, const int32_t* positions, int rows, std::string& err) {
    if (!loaded() || !q || !scales || !positions || rows < 1 || rows > B) { err = "bdraft2: invalid replay"; return false; }
    try {
        const OnDevice on(device_);
        if (!stage_positions(positions, rows, err) ||
            !gpu::copy_async(context_.quantized, q, (size_t) rows * H, stream_, "bdraft2 replay") ||
            !gpu::copy_async(context_.scales, scales, (size_t) rows * H / 512 * sizeof(float), stream_, "bdraft2 replay")) {
            gpu::stream_sync(stream_); err = "bdraft2: replay staging"; return false;
        }
        weights_.context_replay(context_.quantized, context_.scales, rows, positions_, rope_, keys_, values_, SLOTS,
                                tags_, epoch_, context_, stream_);
        if (!gpu::stream_sync(stream_)) throw std::runtime_error("bdraft2: replay");
        return true;
    } catch (const std::exception& e) { gpu::stream_sync(stream_); err = e.what(); return false; }
}

bool Bdraft2Runner::draft(int32_t anchor, int64_t position, int32_t* drafts, float* probabilities, std::string& err) {
    if (!loaded() || !drafts || !probabilities || position < 0 || position > INT32_MAX - B) {
        err = "bdraft2: invalid draft"; return false;
    }
    try {
        const OnDevice on(device_);
        int32_t positions[B];
        for (int i = 0; i < B; ++i) positions[i] = (int32_t) position + i;
        std::vector<float> embedding(H);
        if (!model_->embed_token(anchor, embedding.data()) || !stage_positions(positions, B, err) ||
            !gpu::copy_async(anchor_embedding_, embedding.data(), H * sizeof(float), stream_, "bdraft2 anchor") ||
            !gpu::copy_async(anchor_, &anchor, sizeof anchor, stream_, "bdraft2 anchor"))
            throw std::runtime_error("bdraft2: anchor staging");
        weights_.block(anchor_embedding_, mask_embedding_, anchor_, positions_, rope_, keys_, values_, SLOTS, tags_,
                       epoch_, block_, out_, probabilities_, stream_);
        if (!gpu::copy_async(drafts, out_, (B - 1) * sizeof(int32_t), stream_, "bdraft2 result") ||
            !gpu::copy_async(probabilities, probabilities_, (B - 1) * sizeof(float), stream_, "bdraft2 result") ||
            !gpu::stream_sync(stream_)) throw std::runtime_error("bdraft2: draft result");
        return true;
    } catch (const std::exception& e) { gpu::stream_sync(stream_); err = e.what(); return false; }
}

}  // namespace strata::core
