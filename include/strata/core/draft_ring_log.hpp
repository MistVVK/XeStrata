// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// Committed activation rows and token lineage for a block drafter's ring reconstruction.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace strata::core {

constexpr int kDraftLogChunk = 256;
struct DraftLogChunk { std::vector<int8_t> q; std::vector<float> d; std::vector<uint8_t> ok; };
struct DraftLineage {
    size_t wq = 2560;   ///< int8 codes per row, with wq / 32 scales
    std::vector<int32_t> hist;
    std::vector<std::shared_ptr<DraftLogChunk>> chunks;
    size_t scale_group = 32;   ///< one scale per this many codes (yskn: 512)
    bool logged(int64_t pos) const;
    uint64_t bytes() const;
    void truncate(int64_t keep);
    DraftLogChunk& writable(size_t chunk);
};

}  // namespace strata::core
