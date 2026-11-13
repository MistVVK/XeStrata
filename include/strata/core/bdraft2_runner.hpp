// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Buffer ownership and stream execution for the YASAKANI block drafter.
#pragma once

#include "strata/core/bdraft2.hpp"
#include "strata/core/gpu.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {
class PlainModel;

class Bdraft2Runner {
public:
    Bdraft2Runner() = default;
    ~Bdraft2Runner();
    Bdraft2Runner(const Bdraft2Runner&) = delete;
    Bdraft2Runner& operator=(const Bdraft2Runner&) = delete;
    // Read the target tap list before PlainModel::load, which allocates its tap staging then.
    static bool taps(const std::string& path, std::vector<int>& result, std::string& err);
    bool load(const std::string& path, const std::string& head, PlainModel& model, int device, std::string& err);
    void release();
    bool loaded() const { return weights_.loaded(); }
    uint64_t bytes() const { return bytes_ + weights_.bytes(); }
    void set_head_mode(Bdraft2Weights::HeadMode mode) { mode_ = mode; weights_.set_head_mode(mode); }
    bool head_auto() const { return mode_ == Bdraft2Weights::HeadMode::Auto; }
    void set_probe_shead(bool use) { weights_.set_probe_shead(use); }
    bool has_shead() const { return weights_.has_shead(); }
    bool uses_shead() const { return weights_.uses_shead(); }
    // The caller first selects SHEAD. This runner uses eager launches, so draining the stream releases every
    // reference to the full-head scratch; a future graph owner must destroy those graphs before this call.
    bool release_full_head(std::string& err);
    // These update only committed positions. q and scales receive the exact rows to put into the shared ring log.
    bool context(const float* taps, int rows, int64_t pos0, std::vector<int8_t>& q, std::vector<float>& scales,
                 std::string& err);
    // Restore the saved rows at their original positions, without recomputing the context projection.
    bool replay(const int8_t* q, const float* scales, const int32_t* positions, int rows, std::string& err);
    // Drafts after the supplied anchor. The caller first commits the previous window's taps with context().
    bool draft(int32_t anchor, int64_t position, int32_t* drafts, float* probabilities, std::string& err);
    // Adapt the shared history to the runner without duplicating its begin/stash/rewind policy. The epoch is set
    // both before a replay and after begin: a cache hit may not need replay, and a failed replay invalidates it again.
    template <class History>
    bool begin_request(History& history, const int32_t* ids, int64_t n, int64_t resume, bool* cold,
                       std::string& replay_error, std::string& err) {
        auto restore = [&](const int32_t* pos, const int8_t* q, const float* scales, int rows, int32_t ep,
                           std::string& why) {
            return epoch(ep, why) && replay(q, scales, pos, rows, why);
        };
        if (!history.begin_request(ids, n, resume, restore, cold, replay_error)) {
            err = replay_error;
            return false;
        }
        return epoch(history.epoch(), err);
    }
    // Record exactly the activation rows used by the committed context update. Token history is a separate
    // history.note_rows() call, as for DFlash2; a failure must never put a partial row in the log.
    template <class History>
    bool commit_context(History& history, const float* taps, int rows, int64_t pos0, std::string& err) {
        std::vector<int8_t> q;
        std::vector<float> scales;
        if (!epoch(history.epoch(), err) || !context(taps, rows, pos0, q, scales, err)) return false;
        history.commit(pos0, rows, q.data(), scales.data());
        return true;
    }
    // A new lineage changes the epoch; the shared history decides when this is required.
    bool epoch(int32_t value, std::string& err);

private:
    bool stage_positions(const int32_t* positions, int rows, std::string& err);
    Bdraft2Weights weights_;
    Bdraft2Weights::HeadMode mode_ = Bdraft2Weights::HeadMode::Auto;
    Bdraft2Weights::FullHead full_;
    uint64_t full_bytes_ = 0;
    Bdraft2Weights::ContextWorkspace context_;
    Bdraft2Weights::BlockWorkspace block_;
    PlainModel* model_ = nullptr;
    gpu::Stream stream_ = nullptr;
    int device_ = -1;
    uint64_t bytes_ = 0;
    std::vector<void*> allocations_;
    float* keys_[5] = {};
    float* values_[5] = {};
    int32_t *positions_ = nullptr, *anchor_ = nullptr, *epoch_ = nullptr, *out_ = nullptr;
    int64_t* tags_ = nullptr;
    float *rope_ = nullptr, *anchor_embedding_ = nullptr, *mask_embedding_ = nullptr, *probabilities_ = nullptr;
};

}  // namespace strata::core
