// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Committed activation rows and token lineage for a block drafter's ring reconstruction.
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <string>
#include <memory>
#include <vector>

namespace strata::core {

constexpr int kDraftLogChunk = 256;
struct DraftLogChunk { std::vector<int8_t> q; std::vector<float> d; std::vector<uint8_t> ok; };
struct DraftLineage {
    size_t wq = 2560;   ///< int8 codes per row, with wq / 32 scales
    std::vector<int32_t> hist;
    std::vector<std::shared_ptr<DraftLogChunk>> chunks;
    size_t scale_group = 32;   ///< one scale per this many codes
    bool logged(int64_t pos) const;
    uint64_t bytes() const;
    void truncate(int64_t keep);
    DraftLogChunk& writable(size_t chunk);
};

// Common ring state machine. Replay receives the missing committed rows in position order, with q [rows][width]
// and scales [rows][width / scale_group], at most 8 rows per call. It reconstructs the drafter's K/V with the supplied
// epoch. The callback must finish reading these host buffers before returning. After begin_request(), the caller also applies epoch() to its device,
// even when replay was not needed; only the caller knows how its device stores the epoch.
class DraftRingHistory {
public:
    using Replay = std::function<bool(const int32_t* positions, const int8_t* q, const float* scales, int rows,
                                      int32_t epoch, std::string& err)>;
    struct Stats {
        int64_t requests = 0, rebuilds = 0, rows_rebuilt = 0, rows_missing = 0, stash_hits = 0, stash_stores = 0,
                stash_evictions = 0, cold_starts = 0;
        double ms_rebuild = 0, ms_swap = 0;
    };
    void init(size_t width, size_t scale_group, int slots, int window);
    void configure(bool enabled, uint64_t stash_budget) { enabled_ = enabled; stash_budget_ = stash_budget; }
    void reset();
    bool begin_request(const int32_t* ids, int64_t n, int64_t resume, const Replay& replay, bool* cold, std::string& replay_error);
    void note_rows(const int32_t* toks, int64_t n, int64_t pos0);
    // After an update was enqueued, record its slots; once its quantized rows are on the host, store them.
    void mark_slots(int64_t pos0, int rows);
    // Commit the exact host activation rows returned by a drafter's context update.
    void commit(int64_t pos0, int rows, const int8_t* q, const float* scales) {
        mark_slots(pos0, rows); store(pos0, rows, q, scales); mark_end(pos0 + rows);
    }
    void store(int64_t pos0, int rows, const int8_t* q, const float* scales);
    void mark_end(int64_t end) { if (end > ring_hi_) ring_hi_ = end; }
    bool enabled() const { return enabled_; }
    int32_t epoch() const { return epoch_; }
    uint64_t log_bytes() const { return current_.bytes(); }
    uint64_t stash_bytes() const;
    size_t stash_size() const { return stash_.size(); }
    size_t history_size() const { return current_.hist.size(); }
    Stats stats;
    int64_t last_rebuilt = 0, last_missing = 0;
    double last_ms = 0;
    bool last_swapped = false;

private:
    void invalidate();
    void stash_push(DraftLineage&& lineage);
    bool rebuild(int64_t lo, int64_t hi, const Replay& replay, int64_t& rebuilt, int64_t& missing, std::string& err);
    size_t width_ = 2560, scale_group_ = 32;
    int slots_ = 0, window_ = 0;
    bool enabled_ = true;
    uint64_t stash_budget_ = 256ull << 20;
    DraftLineage current_;
    std::deque<DraftLineage> stash_;
    std::vector<int64_t> slot_pos_;
    int64_t ring_hi_ = 0;
    int32_t epoch_ = 0;
};

}  // namespace strata::core
