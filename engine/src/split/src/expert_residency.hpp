// Choose cold experts for RAM without taking a size pool below the cache's window floor.
#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fnx {

struct ExpertResidency {
    std::vector<std::vector<char>> resident;
    double moved = 0;
    int count = 0;
};

// sizes are whole experts' (gate|up, down) bytes. Every card scales both by its FF slice, so these
// are the same pools Up::expert_cache builds from the sliced strides, including unequal card splits.
inline ExpertResidency expert_residency(const std::vector<std::pair<size_t, size_t>>& sizes,
                                      const std::vector<double>& score, int nexp, int floor,
                                      double want, int extra) {
    std::vector<std::pair<size_t, size_t>> pools;
    std::vector<int> pool_of, left;
    for (const auto& size : sizes) {
        auto it = std::find(pools.begin(), pools.end(), size);
        if (it == pools.end()) {
            pools.push_back(size);
            left.push_back(0);
            it = pools.end() - 1;
        }
        const int p = (int) (it - pools.begin());
        pool_of.push_back(p);
        left[(size_t) p] += nexp;
    }
    for (int n : left)
        if (n < floor) throw std::runtime_error("expert residency: a size pool has fewer experts than a window needs");
    std::vector<int> order(sizes.size() * (size_t) nexp);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return score[(size_t) a] < score[(size_t) b]; });
    ExpertResidency plan;
    plan.resident.assign(sizes.size(), std::vector<char>((size_t) nexp, 1));
    int added = 0;
    for (int i : order) {
        if (plan.moved >= want && added >= extra) break;
        const size_t l = (size_t) (i / nexp), p = (size_t) pool_of[l];
        if (left[p] == floor) continue;
        if (plan.moved >= want) ++added;
        plan.resident[l][(size_t) (i % nexp)] = 0;
        --left[p];
        plan.moved += (double) sizes[l].first + sizes[l].second;
        ++plan.count;
    }
    if (plan.moved < want || added < extra)
        throw std::runtime_error("expert residency: requested RAM placement leaves too few slots for a window in every size pool");
    return plan;
}

}  // namespace fnx
