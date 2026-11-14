// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// The YD1S unigram and bigram draft table, after YASAKANI's specx.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::spec {

class StaticDrafter {
public:
    // The file contains token IDs, not a tokenizer identity. The caller must select the table for this model's
    // tokenizer, even when another tokenizer has the same vocabulary size. A failed load leaves this object unchanged.
    bool load(const std::string& path, int32_t vocab, std::string& err);
    // Follow bigrams before unigrams. -1 means there is no proposal. prev may be -1 at the beginning of a sequence.
    int32_t next(int32_t prev, int32_t cur) const;
    int propose(int32_t prev, int32_t cur, int max, int32_t* out) const;
    bool loaded() const { return !uni_.empty(); }
    // XESTRATA_DF2_STATIC=auto|0|1 and XESTRATA_DF2_STATIC_TABLE=path. Auto offers the table to the startup probe.
    // Returns -1 for auto, 0 for off, 1 for forced on; throws on invalid settings or an invalid file.
    int configure(int32_t vocab);

private:
    std::vector<int32_t> uni_;
    std::unordered_map<uint64_t, int32_t> bi_;
};

}  // namespace strata::spec
