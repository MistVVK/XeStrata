// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2026 the YASAKANI authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND Apache-2.0
// A model-specific SHEAD001 output head. The caller supplies matching drafter states and tokenizer.
#pragma once

#include <cstdint>
#include <string>

namespace strata::core {

class Shead {
public:
    Shead() = default;
    ~Shead();
    Shead(const Shead&) = delete;
    Shead& operator=(const Shead&) = delete;
    // Fail without a live head on malformed data, allocation or transfer failure. max_rows is the drafter's capacity.
    bool load(const std::string& path, int device, int vocab, int hidden, int max_rows, std::string& err);
    // The caller must finish work on its stream before release or destruction.
    void release();
    // h [rows][hidden], top_id and top_value [rows][16], all device buffers.
    void top512(const float* h, int rows, int32_t* top_id, float* top_value, void* stream);
    bool loaded() const { return bt_ != nullptr; }
    uint64_t bytes() const { return bytes_; }

private:
    int device_ = -1, vocab_ = 0, hidden_ = 0, rank_ = 0, max_rows_ = 0;
    uint64_t bytes_ = 0;
    float *bt_ = nullptr, *as_ = nullptr, *ws_ = nullptr, *q_ = nullptr, *score_ = nullptr, *exact_ = nullptr;
    int8_t *a_ = nullptr, *w_ = nullptr;
    int32_t* candidates_ = nullptr;
    int32_t* sort_tmp_ = nullptr;
};

}  // namespace strata::core
