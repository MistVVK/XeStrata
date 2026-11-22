#pragma once

#include <cstdint>
#include <vector>

namespace strata::prefill::bmg_window {
struct PairPiece {
    int rows = 0;
    float *res = nullptr, *shared = nullptr, *weight = nullptr, *inj = nullptr, *sg = nullptr;
    uint16_t* mixed = nullptr;
    std::vector<int32_t> count, off, ids, source, steps;
    int64_t step_position = -1, step_rows = 0;
};

struct Slice {
    int piece, rows, source, target = 0;
};
struct Product {
    int expert;
    std::vector<Slice> slices;
    bool merge_gate = false, merge_down = false;
    int slot = -1, job = -1;
    const uint8_t* pinned = nullptr;
};
enum class BatchKind { Dense, Int8, Packed };
struct Batch {
    BatchKind kind = BatchKind::Dense;
    int variant = 0;
    std::vector<Product> experts;
};

// oneMKL's gate/up and down reductions have separate row-dependent orders.
// A projection may combine slices only if each input and their sum use the same order.
struct MergeProducts {
    bool gate, down;
};
constexpr int gate_order(int rows) {
    if (rows <= 32 || (rows >= 97 && rows <= 192))
        return 4;
    return rows <= 768 ? 2 : 1;
}
constexpr bool down_order(int rows) { return rows >= 129 && rows <= 192; }
constexpr MergeProducts merge_products(int left, int right) {
    if (left <= 0 || right <= 0)
        return {false, false};
    const int sum = left + right;
    return {gate_order(left) == gate_order(right) && gate_order(left) == gate_order(sum),
            down_order(left) == down_order(right) && down_order(left) == down_order(sum)};
}

std::vector<Batch> plan_batches(const PairPiece (&pieces)[2], int chunk, int gu_type, int i8m, bool can8,
                                double cross, bool use_direct);
} // namespace strata::prefill::bmg_window
