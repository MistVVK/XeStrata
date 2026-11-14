// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// The block drafter's copy-on-write activation lineage, shared by its ring users.
#include "strata/core/draft_ring_log.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strata::core {

bool DraftLineage::logged(int64_t pos) const {
    const size_t c = (size_t) (pos / kDraftLogChunk);
    return pos >= 0 && c < chunks.size() && chunks[c] && chunks[c]->ok[(size_t) (pos % kDraftLogChunk)] != 0;
}

uint64_t DraftLineage::bytes() const {
    uint64_t b = (uint64_t) hist.capacity() * 4;
    for (const auto& c : chunks)
        if (c) b += c->q.size() + c->d.size() * 4 + c->ok.size();
    return b;
}

DraftLogChunk& DraftLineage::writable(size_t chunk) {
    if (chunks.size() <= chunk) chunks.resize(chunk + 1);
    if (!chunks[chunk]) {
        auto c = std::make_shared<DraftLogChunk>();
        c->q.assign((size_t) kDraftLogChunk * wq, 0);
        c->d.assign((size_t) kDraftLogChunk * (wq / scale_group), 0.0f);
        c->ok.assign((size_t) kDraftLogChunk, 0);
        chunks[chunk] = std::move(c);
    } else if (chunks[chunk].use_count() > 1) {
        chunks[chunk] = std::make_shared<DraftLogChunk>(*chunks[chunk]);   // copy on write: a stashed copy keeps the old rows
    }
    return *chunks[chunk];
}

void DraftLineage::truncate(int64_t keep) {
    keep = std::max<int64_t>(0, keep);
    const size_t c0 = (size_t) (keep / kDraftLogChunk);
    for (size_t c = c0 + 1; c < chunks.size(); ++c) chunks[c].reset();
    if (chunks.size() > c0 + 1) chunks.resize(c0 + 1);
    if (c0 < chunks.size() && chunks[c0]) {
        if (keep % kDraftLogChunk == 0) {
            chunks[c0].reset();
        } else {
            DraftLogChunk& ch = writable(c0);
            for (int i = (int) (keep % kDraftLogChunk); i < kDraftLogChunk; ++i) ch.ok[(size_t) i] = 0;
        }
    }
}

void DraftRingHistory::init(size_t width, size_t scale_group, int slots, int window) {
    if (!width || !scale_group || width % scale_group || slots < 1 || window < 1 || window > slots)
        throw std::invalid_argument("draft ring: invalid geometry");
    width_ = width; scale_group_ = scale_group; slots_ = slots; window_ = window;
    slot_pos_.assign((size_t) slots, -1);
    current_ = DraftLineage{width_, {}, {}, scale_group_};
    stash_.clear(); ring_hi_ = 0; epoch_ = 0;
}

void DraftRingHistory::invalidate() {
    ++epoch_;
    ring_hi_ = 0;
    std::fill(slot_pos_.begin(), slot_pos_.end(), -1);
}

void DraftRingHistory::reset() {
    current_ = DraftLineage{width_, {}, {}, scale_group_};
    stash_.clear();
    invalidate();
}

uint64_t DraftRingHistory::stash_bytes() const {
    uint64_t bytes = 0;
    for (const auto& lineage : stash_) bytes += lineage.bytes();
    return bytes;
}

void DraftRingHistory::stash_push(DraftLineage&& lineage) {
    if (!enabled_ || stash_budget_ == 0 || lineage.hist.size() < 128 || lineage.chunks.empty()) return;
    stash_.push_back(std::move(lineage));
    ++stats.stash_stores;
    uint64_t total = stash_bytes();
    while (!stash_.empty() && total > stash_budget_) {
        total -= std::min(total, stash_.front().bytes());
        stash_.pop_front();
        ++stats.stash_evictions;
    }
}

void DraftRingHistory::mark_slots(int64_t pos0, int rows) {
    if (!enabled_) return;
    if (slots_ < 1 || pos0 < 0 || rows < 0) throw std::invalid_argument("draft ring: invalid slot range");
    for (int i = 0; i < rows; ++i) slot_pos_[(size_t) ((pos0 + i) % slots_)] = ((int64_t) epoch_ << 32) | (pos0 + i);
}

void DraftRingHistory::store(int64_t pos0, int rows, const int8_t* q, const float* scales) {
    if (!enabled_) return;
    if (slots_ < 1 || pos0 < 0 || rows < 0 || (rows > 0 && (!q || !scales)))
        throw std::invalid_argument("draft ring: invalid activation rows");
    const size_t d = width_ / scale_group_;
    for (int i = 0; i < rows; ++i) {
        const int64_t pos = pos0 + i;
        auto& chunk = current_.writable((size_t) (pos / kDraftLogChunk));
        const size_t row = (size_t) (pos % kDraftLogChunk);
        std::memcpy(&chunk.q[row * width_], q + (size_t) i * width_, width_);
        std::memcpy(&chunk.d[row * d], scales + (size_t) i * d, d * sizeof(float));
        chunk.ok[row] = 1;
    }
}

bool DraftRingHistory::rebuild(int64_t lo, int64_t hi, const Replay& replay, int64_t& rebuilt, int64_t& missing,
                               std::string& err) {
    rebuilt = missing = 0;
    std::vector<int32_t> pos;
    for (int64_t p = lo; p < hi; ++p) {
        if (slot_pos_[(size_t) (p % slots_)] == (((int64_t) epoch_ << 32) | p)) continue;
        if (current_.logged(p)) pos.push_back((int32_t) p); else ++missing;
    }
    if (pos.empty()) return true;
    const size_t d = width_ / scale_group_;
    std::vector<int8_t> q(pos.size() * width_);
    std::vector<float> scales(pos.size() * d);
    for (size_t i = 0; i < pos.size(); ++i) {
        const size_t p = (size_t) pos[i], r = p % kDraftLogChunk;
        const auto& chunk = *current_.chunks[p / kDraftLogChunk];
        std::memcpy(&q[i * width_], &chunk.q[r * width_], width_);
        std::memcpy(&scales[i * d], &chunk.d[r * d], d * sizeof(float));
    }
    for (size_t off = 0; off < pos.size(); off += 8) {
        const int rows = (int) std::min<size_t>(8, pos.size() - off);
        if (!replay || !replay(pos.data() + off, q.data() + off * width_, scales.data() + off * d, rows, epoch_, err))
            return false;
    }
    for (int32_t p : pos) slot_pos_[(size_t) (p % slots_)] = ((int64_t) epoch_ << 32) | p;
    rebuilt = (int64_t) pos.size();
    return true;
}

bool DraftRingHistory::begin_request(const int32_t* ids, int64_t n, int64_t resume, const Replay& replay, bool* cold,
                             std::string& replay_error) {
    using Clock = std::chrono::steady_clock;
    replay_error.clear();
    if (slots_ < 1 || n < 0 || n > std::numeric_limits<int32_t>::max() || ids == nullptr) {
        replay_error = "draft ring: invalid request";
        return false;
    }
    const auto start = Clock::now();
    resume = std::max<int64_t>(0, std::min(resume, n));
    if (cold) *cold = false;
    ++stats.requests;
    last_rebuilt = last_missing = 0; last_ms = 0; last_swapped = false;
    auto lcp = [&](const std::vector<int32_t>& h) {
        int64_t i = 0, lim = std::min<int64_t>((int64_t) h.size(), resume);
        while (i < lim && h[(size_t) i] == ids[i]) ++i;
        return i;
    };
    if (!enabled_) {
        const int64_t prefix = lcp(current_.hist);
        const bool stale = ring_hi_ > prefix && prefix < resume;
        if (stale) invalidate();
        if (cold) *cold = stale;
        current_.hist.assign(ids, ids + resume);
        return true;
    }
    bool covered = lcp(current_.hist) >= resume, swapped = false;
    if (!covered) {
        for (size_t i = stash_.size(); i-- > 0;) {
            if (lcp(stash_[i].hist) < resume) continue;
            DraftLineage got = std::move(stash_[i]);
            stash_.erase(stash_.begin() + (std::ptrdiff_t) i);
            stash_push(std::move(current_));
            current_ = std::move(got);
            invalidate();
            covered = swapped = true;
            ++stats.stash_hits;
            break;
        }
    }
    if (!covered) {
        stash_push(std::move(current_));
        current_ = DraftLineage{width_, {}, {}, scale_group_};
        invalidate();
        ++stats.cold_starts;
    } else if ((int64_t) current_.hist.size() >= resume + 128) {
        DraftLineage copy = current_;
        stash_push(std::move(copy));
    }
    if ((int64_t) current_.hist.size() > resume) { current_.hist.resize((size_t) resume); current_.truncate(resume); }
    if (!covered) current_.hist.assign(ids, ids + resume);
    int64_t rebuilt = 0, missing = 0;
    const int64_t lo = std::max<int64_t>(0, n - (window_ - 1)), hi = resume;
    if (hi > lo && !rebuild(lo, hi, replay, rebuilt, missing, replay_error)) {
        invalidate(); missing = hi - lo; rebuilt = 0;
    }
    stats.rows_rebuilt += rebuilt; stats.rows_missing += missing;
    if (rebuilt > 0) ++stats.rebuilds;
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    if (rebuilt > 0) stats.ms_rebuild += ms;
    if (swapped) stats.ms_swap += ms;
    if (cold) *cold = missing > 0;
    last_rebuilt = rebuilt; last_missing = missing; last_ms = ms; last_swapped = swapped;
    return true;
}

void DraftRingHistory::note_rows(const int32_t* toks, int64_t n, int64_t pos0) {
    if (pos0 < 0) return;
    if (n < 0 || (n > 0 && toks == nullptr)) throw std::invalid_argument("draft ring: invalid token rows");
    if (n == 0) return;
    if ((int64_t) current_.hist.size() > pos0) {
        current_.hist.resize((size_t) pos0);
        if (enabled_) current_.truncate(pos0);
    }
    if ((int64_t) current_.hist.size() < pos0) return;
    current_.hist.insert(current_.hist.end(), toks, toks + n);
}

}  // namespace strata::core
