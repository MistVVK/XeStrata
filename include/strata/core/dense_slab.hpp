// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <cstddef>
#include <vector>

namespace strata::core {

// A loader's small, long-lived device weights share allocations when a startup probe finds that live allocations
// slow kernel submission. Large weights keep their own allocation. The caller owns those, this object owns the slabs.
class DenseSlab {
public:
    DenseSlab();
    ~DenseSlab();
    DenseSlab(const DenseSlab&) = delete;
    DenseSlab& operator=(const DenseSlab&) = delete;
    void* alloc(size_t bytes);
    bool owns(const void* p) const;

private:
    struct Block { void* base; size_t size, used; };
    std::vector<Block> blocks_;
    size_t block_bytes_ = 0;
    int device_ = -1;
};
}  // namespace strata::core
