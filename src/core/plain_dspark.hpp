// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata {
class GgufModel;
namespace core {

// CPU correctness reference for Inkling's DSpark layout, not the DFlash2 pack or MTP.
class PlainDspark {
public:
    static constexpr int block = 8;
    static constexpr std::array<int, 4> taps{5, 23, 41, 48};
    using Embedding = std::function<bool(int32_t, float*)>;
    using Head = std::function<bool(const float*, float*)>;

    PlainDspark();
    ~PlainDspark();
    bool load(const std::string& path, const GgufModel& target, std::string& err);
    void reset();
    void commit(const float* features);
    bool propose(int32_t anchor, int64_t pos, int count, const Embedding& embedding, const Head& head,
                 std::vector<int32_t>& tokens, std::string& err) const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace core
}  // namespace strata
