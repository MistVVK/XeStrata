// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// The block drafter's copy-on-write activation lineage, shared by its ring users.
#include "strata/core/draft_ring_log.hpp"

#include <algorithm>

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

}  // namespace strata::core
