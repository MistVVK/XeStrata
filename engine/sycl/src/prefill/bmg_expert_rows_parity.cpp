#include "strata/sycl_queue.hpp"
#include "bmg_expert_rows.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

int main() {
    sycl::queue queue(sycl::gpu_selector_v, sycl::property::queue::in_order{});
    const auto device = queue.get_device();
    if (queue.get_backend() != sycl::backend::ext_oneapi_level_zero ||
        strata::intel_gpu_gen(device) != strata::IntelGpuGen::Battlemage) return 77;

    constexpr int tokens = 17, experts = 10, width = 2560, guard = 128;
    std::mt19937 generator(317);
    std::vector<int32_t> slots(tokens * experts);
    std::iota(slots.begin(), slots.end(), 0);
    std::shuffle(slots.begin(), slots.end(), generator);
    std::vector<uint16_t> input(tokens * width);
    for (auto& value : input) value = uint16_t(generator());
    std::vector<uint16_t> expected(slots.size() * width + 2 * guard, 0x7e55), actual(expected.size());

    auto* x = sycl::malloc_device<uint16_t>(input.size(), queue);
    auto* index = sycl::malloc_device<int32_t>(slots.size(), queue);
    auto* y = sycl::malloc_device<uint16_t>(expected.size(), queue);
    queue.memcpy(x, input.data(), input.size() * sizeof(uint16_t));
    queue.memcpy(index, slots.data(), slots.size() * sizeof(int32_t));
    queue.fill(y, uint16_t(0x7e55), expected.size());
    using strata::prefill::bmg_rows::scatter_expert_rows;
    bool ok = !scatter_expert_rows(queue, x, index, y + guard, tokens, 4, width) &&
              !scatter_expert_rows(queue, x, index, y + guard + 1, tokens, experts, width) &&
               scatter_expert_rows(queue, x, index, y + guard, tokens, experts, width);
    queue.memcpy(actual.data(), y, actual.size() * sizeof(uint16_t)).wait_and_throw();
    for (size_t i = 0; i < slots.size(); ++i) {
        std::copy_n(input.data() + (i / experts) * width, width,
                    expected.data() + guard + slots[i] * width);
    }
    ok = ok && actual == expected;

    // Randomly permute the logical expert rows, including an absent expert. Its
    // successor must start at the same prefix. Check the entire unused upper
    // half and both red zones after the compact scatter.
    std::vector<int32_t> starts{0, 11, 11, 54, 123, tokens * experts};
    std::vector<int32_t> ids(slots.size());
    for (size_t i = 0; i < slots.size(); ++i) {
        ids[i] = int(std::upper_bound(starts.begin(), starts.end(), slots[i]) - starts.begin() - 1);
    }
    const size_t size = 2 * slots.size() * width + 2 * guard;
    std::vector<uint16_t> wanted(size, 0x595a), got(size);
    auto* compact = sycl::malloc_device<uint16_t>(size, queue);
    auto* expert_ids = sycl::malloc_device<int32_t>(ids.size(), queue);
    auto* prefix = sycl::malloc_device<int32_t>(starts.size(), queue);
    queue.fill(compact, uint16_t(0x595a), size);
    queue.memcpy(expert_ids, ids.data(), ids.size() * sizeof(int32_t));
    queue.memcpy(prefix, starts.data(), starts.size() * sizeof(int32_t));
    strata::prefill::bmg_rows::copy_expert_rows<true>(queue, x, index, compact + guard, tokens, expert_ids, prefix);
    queue.memcpy(got.data(), compact, size * sizeof(uint16_t)).wait_and_throw();
    for (size_t i = 0; i < slots.size(); ++i) {
        const int row = starts[ids[i]] + slots[i];
        std::copy_n(input.data() + (i / experts) * width, width, wanted.data() + guard + row * width);
    }
    ok = ok && got == wanted;
    sycl::free(prefix, queue);
    sycl::free(expert_ids, queue);
    sycl::free(compact, queue);
    sycl::free(y, queue);
    sycl::free(index, queue);
    sycl::free(x, queue);
    return ok ? 0 : 1;
}
